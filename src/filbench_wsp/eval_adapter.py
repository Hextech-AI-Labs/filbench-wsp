"""Fail-closed local FilBench scoring and cross-condition parity verification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ScoringMode = Literal["strict_full", "partial", "missing_as_zero"]


@dataclass(frozen=True)
class FilBenchTaskSpec:
    benchmark_key: str
    category: str
    metric: str
    canonical_n: int


FILBENCH_SCORED_TASKS: tuple[FilBenchTaskSpec, ...] = (
    FilBenchTaskSpec("balita_tgl_mcf", "Classical NLP", "acc", 35177),
    FilBenchTaskSpec("cebuaner_ceb_mcf", "Classical NLP", "acc", 1310),
    FilBenchTaskSpec("dengue_filipino_fil:_average", "Classical NLP", "acc_norm", 4015),
    FilBenchTaskSpec("firecs_fil_mcf", "Classical NLP", "acc", 7340),
    FilBenchTaskSpec("sib200_ceb_mcf", "Classical NLP", "acc", 99),
    FilBenchTaskSpec("sib200_tgl_mcf", "Classical NLP", "acc", 99),
    FilBenchTaskSpec("tlunifiedner_tgl_mcf", "Classical NLP", "acc", 1579),
    FilBenchTaskSpec("universalner_ceb_mcf", "Classical NLP", "acc", 49),
    FilBenchTaskSpec("universalner_tgl_mcf", "Classical NLP", "acc", 56),
    FilBenchTaskSpec("global_mmlu_all_tgl_mcf:_average", "Cultural Knowledge", "acc", 14042),
    FilBenchTaskSpec("include_tgl_mcf:_average", "Cultural Knowledge", "acc", 500),
    FilBenchTaskSpec("kalahi_tgl_mcf", "Cultural Knowledge", "acc", 150),
    FilBenchTaskSpec(
        "stingraybench_semantic_appropriateness_tgl_mcf",
        "Cultural Knowledge",
        "acc",
        100,
    ),
    FilBenchTaskSpec("belebele_ceb_mcf", "Reading Comprehension", "acc", 900),
    FilBenchTaskSpec("belebele_fil_mcf", "Reading Comprehension", "acc", 900),
    FilBenchTaskSpec("newsphnli_fil_mcf", "Reading Comprehension", "acc", 90000),
    FilBenchTaskSpec("readability_ceb_mcf", "Reading Comprehension", "acc", 350),
    FilBenchTaskSpec("ntrex128_fil", "Generation", "rougeL", 1997),
    FilBenchTaskSpec("tatoeba_ceb", "Generation", "rougeL", 377),
    FilBenchTaskSpec("tatoeba_tgl", "Generation", "rougeL", 2499),
    FilBenchTaskSpec("tico19_tgl", "Generation", "rougeL", 971),
)

TASK_SPEC_BY_KEY = {spec.benchmark_key: spec for spec in FILBENCH_SCORED_TASKS}
GROUP_PREFIX_TO_CANONICAL_KEY = {
    "dengue_filipino_fil:": "dengue_filipino_fil:_average",
    "global_mmlu_all_tgl_mcf:": "global_mmlu_all_tgl_mcf:_average",
    "include_tgl_mcf:": "include_tgl_mcf:_average",
}
CATEGORIES = ("Classical NLP", "Cultural Knowledge", "Reading Comprehension", "Generation")
REQUIRED_NESTED_HASH_FIELDS = (
    "hash_examples",
    "hash_full_prompts",
    "hash_input_tokens",
    "hash_cont_tokens",
)
REQUIRED_SUMMARY_COUNT_FIELDS = (
    "truncated",
    "non_truncated",
    "non_padded",
    "padded",
    "effective_few_shots",
)


@dataclass
class FilBenchScoreReport:
    evaluated_model_name: str = ""
    scoring_mode: ScoringMode = "strict_full"
    is_partial_eval: bool = False
    is_subsampled: bool = False
    max_samples_config: int | None = None
    num_tasks_evaluated: int = 0
    total_canonical_tasks: int = len(FILBENCH_SCORED_TASKS)
    evaluated_tasks: list[str] = field(default_factory=list)
    per_task_scores_pct: dict[str, float] = field(default_factory=dict)
    subtask_scores_pct: dict[str, float] = field(default_factory=dict)
    missing_tasks: list[str] = field(default_factory=list)
    subsampled_tasks: dict[str, dict[str, int]] = field(default_factory=dict)
    doc_counts_by_task: dict[str, dict[str, int]] = field(default_factory=dict)
    dataset_revisions: dict[str, str | None] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    filbench_macro_score: float | None = None
    filbench_instance_weighted_score: float | None = None
    partial_macro_score_debug: float | None = None
    strict_missing_as_zero: bool = False
    source_files_by_task: dict[str, str] = field(default_factory=dict)
    task_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_lighteval_task_key(full_key: str) -> tuple[str, int | None]:
    parts = full_key.split("|")
    if len(parts) >= 3:
        try:
            shots = int(parts[2])
        except ValueError:
            shots = None
        return parts[1], shots
    if len(parts) == 2:
        return parts[1], 0
    return full_key, 0


def resolve_task_spec(benchmark_key: str) -> tuple[FilBenchTaskSpec | None, bool]:
    if benchmark_key in TASK_SPEC_BY_KEY:
        return TASK_SPEC_BY_KEY[benchmark_key], False
    for prefix, canonical_key in GROUP_PREFIX_TO_CANONICAL_KEY.items():
        if benchmark_key.startswith(prefix) and benchmark_key != canonical_key:
            return TASK_SPEC_BY_KEY[canonical_key], True
    return None, False


def _resolve_group_parent_for_benchmark(benchmark_key: str) -> str | None:
    for prefix, canonical_key in GROUP_PREFIX_TO_CANONICAL_KEY.items():
        if benchmark_key.startswith(prefix) and benchmark_key != canonical_key:
            return canonical_key
    return None


def _extract_doc_counts_and_revision(
    full_key: str, benchmark_key: str, config_tasks: dict[str, Any]
) -> tuple[dict[str, int], str | None]:
    parts = full_key.split("|")
    suite_key = f"{parts[0]}|{parts[1]}" if len(parts) >= 2 else full_key
    task_config = (
        config_tasks.get(suite_key)
        or config_tasks.get(full_key)
        or config_tasks.get(benchmark_key)
    )
    if not isinstance(task_config, dict):
        raise ValueError(
            f"Task '{full_key}' is missing its configuration entry in 'config_tasks'. "
            "Refusing to assume full dataset coverage without document counts."
        )
    original = task_config.get("original_num_docs")
    effective = task_config.get("effective_num_docs")
    if not isinstance(original, int) or not isinstance(effective, int) or original <= 0 or effective <= 0:
        raise ValueError(
            f"Task '{full_key}' has invalid doc counts in 'config_tasks': "
            f"original_num_docs={original!r}, effective_num_docs={effective!r}."
        )
    return {"effective": effective, "original": original}, task_config.get("hf_revision")


@dataclass
class ExtractedLightevalFile:
    model_name: str
    max_samples: int | None
    canonical_metrics: dict[str, float]
    subtask_metrics: dict[str, float]
    task_summaries: dict[str, dict[str, Any]]
    doc_counts_by_task: dict[str, dict[str, int]]
    dataset_revisions: dict[str, str | None]


def validate_task_summary_shape(full_key: str, task_summary: Any) -> dict[str, Any]:
    if not isinstance(task_summary, dict):
        raise ValueError(f"Task '{full_key}' is missing a valid dict in 'summary_tasks'.")
    nested_hashes = task_summary.get("hashes")
    if not isinstance(nested_hashes, dict):
        raise ValueError(
            f"Task '{full_key}' summary_tasks entry is missing the nested 'hashes' dict. "
            f"Top-level keys found: {sorted(task_summary.keys())}"
        )
    missing_hashes = [
        key
        for key in REQUIRED_NESTED_HASH_FIELDS
        if key not in nested_hashes or nested_hashes[key] in (None, "")
    ]
    if missing_hashes:
        raise ValueError(
            f"Task '{full_key}' nested 'hashes' dict is missing required keys {missing_hashes}."
        )
    missing_counts = [key for key in REQUIRED_SUMMARY_COUNT_FIELDS if key not in task_summary]
    if missing_counts:
        raise ValueError(
            f"Task '{full_key}' summary_tasks entry is missing count keys {missing_counts}."
        )
    return task_summary


def extract_metrics_from_lighteval_json(
    json_data: dict[str, Any],
    require_summary_hashes: bool = True,
) -> ExtractedLightevalFile:
    results = json_data.get("results")
    if not isinstance(results, dict):
        raise ValueError("Invalid lighteval JSON: missing or non-dict 'results' block.")
    config_general = json_data.get("config_general")
    if not isinstance(config_general, dict):
        raise ValueError("Invalid lighteval JSON: missing or non-dict 'config_general' block.")
    raw_model_name = config_general.get("model_name")
    if not isinstance(raw_model_name, str) or not raw_model_name.strip():
        raise ValueError(
            "Invalid lighteval JSON: 'config_general.model_name' is missing or empty. "
            "Every results file must identify its model checkpoint."
        )
    model_name = raw_model_name.strip()
    raw_max_samples = config_general.get("max_samples")
    if raw_max_samples is not None and not isinstance(raw_max_samples, int):
        raise ValueError(f"Invalid config_general.max_samples: {raw_max_samples!r}")
    summary_tasks = json_data.get("summary_tasks", {})
    config_tasks = json_data.get("config_tasks", {})
    if require_summary_hashes and not isinstance(config_tasks, dict):
        raise ValueError("Invalid lighteval JSON: missing or non-dict 'config_tasks' block.")
    canonical_scores: dict[str, float] = {}
    subtask_scores: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    doc_counts: dict[str, dict[str, int]] = {}
    revisions: dict[str, str | None] = {}
    aggregate_groups: set[str] = set()
    concrete_groups: set[str] = set()

    for full_key, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        benchmark_key, shots = parse_lighteval_task_key(full_key)
        if shots != 0:
            continue
        spec, is_subtask = resolve_task_spec(benchmark_key)
        if spec is None:
            continue
        if spec.metric not in metrics:
            raise KeyError(
                f"Task '{full_key}' (benchmark '{benchmark_key}') ran, but expected metric "
                f"'{spec.metric}' was not found in its results dict! "
                f"Available metric keys: {sorted(metrics.keys())}"
            )
        target = subtask_scores if is_subtask else canonical_scores
        if benchmark_key in target:
            raise ValueError(f"Duplicate 0-shot entries for benchmark '{benchmark_key}'.")
        target[benchmark_key] = float(metrics[spec.metric])
        is_aggregate = benchmark_key.endswith(":_average")
        if is_aggregate:
            aggregate_groups.add(benchmark_key)
        elif is_subtask:
            concrete_groups.add(resolve_task_spec(benchmark_key)[0].benchmark_key)
        if require_summary_hashes and (not is_aggregate or full_key in summary_tasks):
            if full_key not in summary_tasks:
                raise ValueError(f"Task '{full_key}' is present in 'results' but missing from 'summary_tasks'.")
            summaries[benchmark_key] = validate_task_summary_shape(full_key, summary_tasks[full_key])
        elif full_key in summary_tasks and isinstance(summary_tasks[full_key], dict):
            summaries[benchmark_key] = summary_tasks[full_key]
        if require_summary_hashes and not is_aggregate:
            counts, revision = _extract_doc_counts_and_revision(full_key, benchmark_key, config_tasks)
            doc_counts[benchmark_key] = counts
            revisions[benchmark_key] = revision
    for group_key in aggregate_groups:
        if group_key not in concrete_groups:
            raise ValueError(
                f"Group task '{group_key}' has an aggregate result but no concrete "
                "member subtask in results, summary_tasks, and config_tasks."
            )
        member_counts = [
            counts
            for key, counts in doc_counts.items()
            if _resolve_group_parent_for_benchmark(key) == group_key
        ]
        if member_counts:
            doc_counts[group_key] = {
                "effective": sum(item["effective"] for item in member_counts),
                "original": sum(item["original"] for item in member_counts),
            }
    return ExtractedLightevalFile(
        model_name=model_name,
        max_samples=raw_max_samples,
        canonical_metrics=canonical_scores,
        subtask_metrics=subtask_scores,
        task_summaries=summaries,
        doc_counts_by_task=doc_counts,
        dataset_revisions=revisions,
    )


def compute_filbench_scores(
    raw_task_metrics: dict[str, float],
    *,
    subtask_metrics: dict[str, float] | None = None,
    allow_partial: bool = False,
    strict_missing_as_zero: bool = False,
    max_samples_config: int | None = None,
    doc_counts_by_task: dict[str, dict[str, int]] | None = None,
    dataset_revisions: dict[str, str | None] | None = None,
    evaluated_model_name: str = "",
    source_files_by_task: dict[str, str] | None = None,
    task_summaries: dict[str, dict[str, Any]] | None = None,
) -> FilBenchScoreReport:
    if allow_partial and strict_missing_as_zero:
        raise ValueError(
            "allow_partial and strict_missing_as_zero are mutually exclusive; choose at most one."
        )
    subtask_metrics = subtask_metrics or {}
    doc_counts_by_task = doc_counts_by_task or {}
    dataset_revisions = dataset_revisions or {}
    unknown = set(raw_task_metrics) - set(TASK_SPEC_BY_KEY)
    if unknown:
        raise KeyError(f"Unrecognized FilBench task keys in raw_task_metrics: {sorted(unknown)}")
    if not raw_task_metrics and not subtask_metrics:
        raise ValueError("No scored 0-shot FilBench tasks or subtasks were provided.")

    has_complete_task_set = set(raw_task_metrics) == set(TASK_SPEC_BY_KEY)
    subsampled_tasks = {
        key: counts
        for key, counts in doc_counts_by_task.items()
        if counts.get("effective", 0) < counts.get("original", 0)
        or (
            has_complete_task_set
            and key in TASK_SPEC_BY_KEY
            and not key.endswith(":_average")
            and counts.get("effective", 0) < TASK_SPEC_BY_KEY[key].canonical_n
        )
    }
    is_subsampled = max_samples_config is not None or bool(subsampled_tasks)
    effective_metrics = dict(raw_task_metrics)
    used_subtask_fallback = False
    for prefix, canonical_key in GROUP_PREFIX_TO_CANONICAL_KEY.items():
        if canonical_key not in effective_metrics:
            values = [value for key, value in subtask_metrics.items() if key.startswith(prefix)]
            if values:
                if not allow_partial:
                    raise ValueError(
                        f"Group task '{canonical_key}' is missing its aggregate score; "
                        "found only individual subtask(s). Pass allow_partial=True for subtask smoke tests."
                    )
                effective_metrics[canonical_key] = sum(values) / len(values)
                used_subtask_fallback = True
    if not effective_metrics:
        raise ValueError("No scored 0-shot FilBench tasks or subtasks were provided.")
    if is_subsampled and not allow_partial:
        raise ValueError(
            "Refusing to score subsampled run as a full FilBench "
            f"(max_samples={max_samples_config}, subsampled_tasks={subsampled_tasks}). "
            "Pass allow_partial=True for subsampled smoke/dev runs."
        )
    scoring_mode: ScoringMode = (
        "partial" if allow_partial else
        "missing_as_zero" if strict_missing_as_zero else
        "strict_full"
    )
    evaluated_tasks = sorted(set(raw_task_metrics) | set(subtask_metrics))
    subtask_scores_pct = {key: round(value * 100.0, 4) for key, value in subtask_metrics.items()}
    per_task: dict[str, float] = {}
    missing: list[str] = []
    for spec in FILBENCH_SCORED_TASKS:
        if spec.benchmark_key in effective_metrics:
            per_task[spec.benchmark_key] = round(effective_metrics[spec.benchmark_key] * 100.0, 4)
        else:
            missing.append(spec.benchmark_key)
            if strict_missing_as_zero:
                per_task[spec.benchmark_key] = 0.0
    partial = is_subsampled or used_subtask_fallback or (bool(missing) and scoring_mode == "partial")
    if missing and scoring_mode == "strict_full":
        raise ValueError(
            f"Refusing to compute full FilBench score: {len(missing)} of 21 tasks are missing: "
            f"{missing}. Pass allow_partial=True for a flagged partial run, or "
            "strict_missing_as_zero=True to explicitly score missing tasks as 0."
        )

    category_scores: dict[str, float] = {}
    weighted_sum = 0.0
    weighted_denominator = 0
    for category in CATEGORIES:
        specs = [spec for spec in FILBENCH_SCORED_TASKS if spec.category == category]
        category_sum = 0.0
        category_denominator = 0
        for spec in specs:
            if spec.benchmark_key in effective_metrics:
                value = effective_metrics[spec.benchmark_key] * 100.0
                category_sum += spec.canonical_n * value
                category_denominator += spec.canonical_n
            elif strict_missing_as_zero:
                category_denominator += spec.canonical_n
        if category_denominator:
            category_scores[category] = round(category_sum / category_denominator, 4)
        weighted_sum += category_sum
        weighted_denominator += category_denominator

    macro = round(sum(category_scores.values()) / len(category_scores), 4) if category_scores else 0.0
    micro = round(weighted_sum / weighted_denominator, 4) if weighted_denominator else 0.0
    return FilBenchScoreReport(
        evaluated_model_name=evaluated_model_name,
        scoring_mode=scoring_mode,
        is_partial_eval=partial,
        is_subsampled=is_subsampled,
        max_samples_config=max_samples_config,
        num_tasks_evaluated=len(evaluated_tasks),
        per_task_scores_pct=per_task,
        subtask_scores_pct=subtask_scores_pct,
        evaluated_tasks=evaluated_tasks,
        missing_tasks=missing,
        subsampled_tasks=subsampled_tasks,
        doc_counts_by_task=doc_counts_by_task,
        dataset_revisions=dataset_revisions,
        category_scores=category_scores,
        filbench_macro_score=None if partial else macro,
        filbench_instance_weighted_score=None if partial else micro,
        partial_macro_score_debug=macro if partial else None,
        strict_missing_as_zero=strict_missing_as_zero,
        source_files_by_task=source_files_by_task or {},
        task_summaries=task_summaries or {},
    )


def score_results_directory(
    results_dir: Path,
    *,
    allow_partial: bool = False,
    strict_missing_as_zero: bool = False,
    require_summary_hashes: bool = True,
) -> FilBenchScoreReport:
    if allow_partial and strict_missing_as_zero:
        raise ValueError(
            "allow_partial and strict_missing_as_zero are mutually exclusive; choose at most one."
        )
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    json_files = sorted(results_dir.rglob("results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No lighteval results_*.json files found in {results_dir}")

    metrics: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    models: dict[str, str] = {}
    subtasks: dict[str, float] = {}
    doc_counts: dict[str, dict[str, int]] = {}
    revisions: dict[str, str | None] = {}
    max_samples_config: int | None = None
    for json_file in json_files:
        relative = json_file.relative_to(results_dir).as_posix()
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        extracted = extract_metrics_from_lighteval_json(
            payload, require_summary_hashes=require_summary_hashes
        )
        models[relative] = extracted.model_name
        if extracted.max_samples is not None:
            max_samples_config = extracted.max_samples
        for task, score in extracted.canonical_metrics.items():
            if task in metrics:
                raise ValueError(
                    f"Ambiguous duplicate results for task '{task}' found in both "
                    f"'{sources[task]}' and '{relative}'. Refusing to guess which file wins!"
                )
            metrics[task] = score
            sources[task] = relative
            if task in extracted.task_summaries:
                summaries[task] = extracted.task_summaries[task]
        for task, score in extracted.subtask_metrics.items():
            if task in subtasks:
                raise ValueError(f"Ambiguous duplicate results for subtask '{task}'.")
            subtasks[task] = score
            sources[task] = relative
        summaries.update(extracted.task_summaries)
        doc_counts.update(extracted.doc_counts_by_task)
        revisions.update(extracted.dataset_revisions)

    unique_models = set(models.values())
    if len(unique_models) > 1:
        raise ValueError(
            f"Multiple model_name values detected inside '{results_dir}': {models}. "
            "Refusing to blend results from different checkpoints!"
        )
    return compute_filbench_scores(
        metrics,
        subtask_metrics=subtasks,
        allow_partial=allow_partial,
        strict_missing_as_zero=strict_missing_as_zero,
        max_samples_config=max_samples_config,
        doc_counts_by_task=doc_counts,
        dataset_revisions=revisions,
        evaluated_model_name=next(iter(unique_models), ""),
        source_files_by_task=sources,
        task_summaries=summaries,
    )


def parse_lighteval_results_dir(
    results_dir: Path,
    *,
    allow_partial: bool = False,
    allow_missing_as_zero: bool = False,
) -> dict[str, Any]:
    """Parse a lighteval results directory using the local scoring adapter."""
    return score_results_directory(
        results_dir,
        allow_partial=allow_partial,
        strict_missing_as_zero=allow_missing_as_zero,
    ).to_dict()


def _get_actually_ran_tasks(report: FilBenchScoreReport) -> set[str]:
    """Avoid treating missing-as-zero placeholders as evaluated tasks."""
    if report.source_files_by_task:
        tasks = set(report.source_files_by_task)
    else:
        tasks = set(report.evaluated_tasks)
    return {task for task in tasks if not task.endswith(":_average")}


def verify_cross_condition_input_parity(
    baseline_report: FilBenchScoreReport,
    comparison_report: FilBenchScoreReport,
    *,
    expect_identical_outputs: bool = False,
) -> list[str]:
    """Verify input parity, plus continuation parity when outputs must match."""
    errors: list[str] = []
    baseline_tasks = _get_actually_ran_tasks(baseline_report)
    comparison_tasks = _get_actually_ran_tasks(comparison_report)
    if not baseline_tasks:
        errors.append("Baseline report has 0 actually-ran tasks.")
    if not comparison_tasks:
        errors.append("Comparison report has 0 actually-ran tasks.")
    if not baseline_tasks or not comparison_tasks:
        return errors

    vanished = sorted(baseline_tasks - comparison_tasks)
    extra = sorted(comparison_tasks - baseline_tasks)
    if vanished:
        errors.append(f"Tasks that ran in baseline but vanished in comparison run: {vanished}")
    if extra:
        errors.append(f"Tasks that ran in comparison run but were absent in baseline: {extra}")

    for task in sorted(baseline_tasks & comparison_tasks):
        baseline_docs = baseline_report.doc_counts_by_task.get(task)
        comparison_docs = comparison_report.doc_counts_by_task.get(task)
        if baseline_docs != comparison_docs:
            errors.append(
                f"Task '{task}' mismatch on doc counts: "
                f"baseline={baseline_docs} vs comparison={comparison_docs}"
            )
        baseline_summary = baseline_report.task_summaries.get(task)
        comparison_summary = comparison_report.task_summaries.get(task)
        if not isinstance(baseline_summary, dict):
            errors.append(f"Task '{task}': missing summary_tasks entry in baseline report.")
            continue
        if not isinstance(comparison_summary, dict):
            errors.append(f"Task '{task}': missing summary_tasks entry in comparison report.")
            continue
        baseline_hashes = baseline_summary.get("hashes")
        comparison_hashes = comparison_summary.get("hashes")
        if not isinstance(baseline_hashes, dict):
            errors.append(f"Task '{task}': baseline summary_tasks is missing nested 'hashes' dict.")
            continue
        if not isinstance(comparison_hashes, dict):
            errors.append(f"Task '{task}': comparison summary_tasks is missing nested 'hashes' dict.")
            continue
        input_hash_fields = (
            "hash_examples",
            "hash_full_prompts",
            "hash_input_tokens",
        )
        for field in input_hash_fields:
            if not baseline_hashes.get(field):
                errors.append(f"Task '{task}': baseline missing nested hashes['{field}'].")
            if not comparison_hashes.get(field):
                errors.append(f"Task '{task}': comparison missing nested hashes['{field}'].")
            if baseline_hashes.get(field) and comparison_hashes.get(field) and baseline_hashes[field] != comparison_hashes[field]:
                errors.append(
                    f"Task '{task}' mismatch on hashes['{field}']: "
                    f"baseline={baseline_hashes[field]} vs comparison={comparison_hashes[field]}"
                )
        metric_spec, _ = resolve_task_spec(task)
        check_continuations = expect_identical_outputs or (
            metric_spec is not None and metric_spec.metric in ("acc", "acc_norm")
        )
        continuation_fields = ("hash_cont_tokens",) if check_continuations else ()
        for field in continuation_fields:
            if not baseline_hashes.get(field):
                errors.append(f"Task '{task}': baseline missing nested hashes['{field}'].")
            if not comparison_hashes.get(field):
                errors.append(f"Task '{task}': comparison missing nested hashes['{field}'].")
            if (
                baseline_hashes.get(field)
                and comparison_hashes.get(field)
                and baseline_hashes[field] != comparison_hashes[field]
            ):
                errors.append(
                    f"Task '{task}' mismatch on hashes['{field}']: "
                    f"baseline={baseline_hashes[field]} vs comparison={comparison_hashes[field]}"
                )
        count_fields = REQUIRED_SUMMARY_COUNT_FIELDS if check_continuations else (
            "truncated",
            "non_truncated",
        )
        for field in count_fields:
            if field not in baseline_summary:
                errors.append(f"Task '{task}': baseline missing summary_tasks['{field}'].")
            if field not in comparison_summary:
                errors.append(f"Task '{task}': comparison missing summary_tasks['{field}'].")
            if field in baseline_summary and field in comparison_summary and baseline_summary[field] != comparison_summary[field]:
                errors.append(
                    f"Task '{task}' mismatch on summary_tasks['{field}']: "
                    f"baseline={baseline_summary[field]} vs comparison={comparison_summary[field]}"
                )
    return errors


def compute_macro_score(category_scores: dict[str, float]) -> float:
    """Backward-compatible unweighted category macro score."""
    if not category_scores:
        raise ValueError("category_scores must contain at least one category")
    if any(not isinstance(value, (int, float)) for value in category_scores.values()):
        raise TypeError("category scores must be numeric")
    return sum(category_scores.values()) / len(category_scores)
