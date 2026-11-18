"""Fail-closed paired statistics for lighteval detail Parquet artifacts."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pyarrow.parquet as pq

from filbench_wsp.eval_adapter import (
    GROUP_PREFIX_TO_CANONICAL_KEY,
    TASK_SPEC_BY_KEY,
)

_DETAILS_FILENAME_RE = re.compile(
    r"^details_filbench[|_](?P<raw_task>.+?)[|_](?P<shots>\d+)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T.+)\.parquet$"
)


def resolve_concrete_task_from_parquet_name(filename: str) -> Tuple[str, str, str]:
    """Resolve a details filename to concrete task, canonical task, and metric."""
    match = _DETAILS_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized lighteval details Parquet filename: {filename!r}")
    raw_task = match.group("raw_task")

    if raw_task in TASK_SPEC_BY_KEY and not raw_task.endswith(":_average"):
        return raw_task, raw_task, TASK_SPEC_BY_KEY[raw_task].metric

    for prefix, canonical_key in GROUP_PREFIX_TO_CANONICAL_KEY.items():
        if raw_task.startswith(prefix) and raw_task != canonical_key:
            return raw_task, canonical_key, TASK_SPEC_BY_KEY[canonical_key].metric

    for prefix, canonical_key in GROUP_PREFIX_TO_CANONICAL_KEY.items():
        underscore_prefix = prefix[:-1] + "_"
        if raw_task.startswith(underscore_prefix):
            subtask = raw_task[len(underscore_prefix) :]
            if subtask:
                return (
                    f"{prefix}{subtask}",
                    canonical_key,
                    TASK_SPEC_BY_KEY[canonical_key].metric,
                )

    raise ValueError(
        f"Parquet task {raw_task!r} does not map to a scored FilBench task."
    )


def load_task_parquet_examples(
    parquet_path: Path, expected_metric: str
) -> List[Dict[str, Any]]:
    """Load aligned example identity and percentage score from one detail file."""
    table = pq.read_table(parquet_path, columns=["example", "full_prompt", "metrics"])
    columns = table.to_pydict()
    examples = columns.get("example", [])
    prompts = columns.get("full_prompt", [])
    metrics = columns.get("metrics", [])
    if not (len(examples) == len(prompts) == len(metrics)):
        raise ValueError(f"Column length mismatch in {parquet_path}")

    rows: List[Dict[str, Any]] = []
    for index, metric_row in enumerate(metrics):
        if not isinstance(metric_row, dict) or expected_metric not in metric_row:
            raise ValueError(
                f"Row {index} in {parquet_path} is missing metric {expected_metric!r}."
            )
        value = metric_row[expected_metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Row {index} in {parquet_path} has non-numeric metric "
                f"{expected_metric!r}: {value!r}"
            )
        rows.append(
            {
                "row_index": index,
                "example": str(examples[index]),
                "full_prompt": str(prompts[index]),
                "score_pct": float(value) * 100.0,
            }
        )
    return rows


def discover_details_parquets(
    results_dir: Path,
) -> Dict[str, Tuple[Path, str, str]]:
    """Discover unique scored concrete-task detail files recursively."""
    discovered: Dict[str, Tuple[Path, str, str]] = {}
    for parquet_path in sorted(results_dir.rglob("details_*.parquet")):
        concrete, canonical, metric = resolve_concrete_task_from_parquet_name(
            parquet_path.name
        )
        if concrete in discovered:
            raise ValueError(
                f"Duplicate details Parquet for concrete task {concrete!r}: "
                f"{discovered[concrete][0]} and {parquet_path}"
            )
        discovered[concrete] = (parquet_path, canonical, metric)
    return discovered


def exact_sign_test_two_sided_p(n_wins: int, n_losses: int) -> float:
    """Return the exact two-sided binomial sign-test p-value."""
    discordant = n_wins + n_losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n_wins, n_losses) + 1))
    return min(1.0, 2.0 * tail * (0.5**discordant))


def paired_bootstrap_ci(
    deltas: Sequence[float],
    num_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return bootstrap standard error and percentile confidence interval."""
    if not deltas:
        raise ValueError("Cannot compute paired bootstrap CI on empty deltas.")
    if num_resamples <= 0 or not 0.0 < ci_level < 1.0:
        raise ValueError("num_resamples must be positive and ci_level must be in (0, 1).")
    if len(deltas) == 1 or all(abs(value - deltas[0]) <= 1e-12 for value in deltas):
        value = float(deltas[0])
        return 0.0, value, value

    rng = random.Random(seed)
    means = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(num_resamples)
    ]
    means.sort()
    alpha = 1.0 - ci_level
    low = max(0, min(num_resamples - 1, math.floor(alpha * num_resamples / 2.0)))
    high = max(
        0,
        min(num_resamples - 1, math.ceil((1.0 - alpha / 2.0) * num_resamples) - 1),
    )
    mean = sum(means) / num_resamples
    variance = sum((value - mean) ** 2 for value in means) / max(1, num_resamples - 1)
    return math.sqrt(variance), means[low], means[high]


def compute_paired_series_stats(
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    task_label: str,
    num_resamples: int = 2000,
    seed: int = 42,
    tie_tol: float = 1e-9,
) -> Dict[str, Any]:
    """Validate row alignment and compute paired improvement statistics."""
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            f"Per-example row count mismatch on {task_label!r}: "
            f"baseline has {len(baseline_rows)} rows vs candidate {len(candidate_rows)} rows."
        )
    if not baseline_rows:
        raise ValueError(f"Zero per-example rows found for {task_label!r}.")

    deltas: List[float] = []
    wins = losses = 0
    for index, (baseline, candidate) in enumerate(zip(baseline_rows, candidate_rows)):
        if (
            baseline["full_prompt"] != candidate["full_prompt"]
            or baseline["example"] != candidate["example"]
        ):
            raise ValueError(
                f"Per-example prompt alignment mismatch on {task_label!r} at row {index}."
            )
        delta = float(candidate["score_pct"]) - float(baseline["score_pct"])
        deltas.append(delta)
        if delta > tie_tol:
            wins += 1
        elif delta < -tie_tol:
            losses += 1

    ties = len(deltas) - wins - losses
    bootstrap_se, ci_low, ci_high = paired_bootstrap_ci(
        deltas, num_resamples=num_resamples, seed=seed
    )
    return {
        "n_examples": len(deltas),
        "baseline_mean_pct": round(sum(row["score_pct"] for row in baseline_rows) / len(deltas), 6),
        "candidate_mean_pct": round(sum(row["score_pct"] for row in candidate_rows) / len(deltas), 6),
        "mean_delta_pct": round(sum(deltas) / len(deltas), 6),
        "bootstrap_se_pct": round(bootstrap_se, 6),
        "ci95_low_pct": round(ci_low, 6),
        "ci95_high_pct": round(ci_high, 6),
        "n_improved": wins,
        "n_regressed": losses,
        "n_unchanged": ties,
        "sign_test_p_two_sided": round(exact_sign_test_two_sided_p(wins, losses), 6),
    }


def compare_experiment_details_dirs(
    baseline_results_dir: Path,
    candidate_results_dir: Path,
    num_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pair every concrete task in two result trees and compute grouped statistics."""
    baseline = discover_details_parquets(baseline_results_dir)
    candidate = discover_details_parquets(candidate_results_dir)
    if set(baseline) != set(candidate):
        raise ValueError(
            "Concrete details Parquet task set mismatch: "
            f"only_in_baseline={sorted(set(baseline) - set(candidate))}, "
            f"only_in_candidate={sorted(set(candidate) - set(baseline))}"
        )
    if not baseline:
        raise FileNotFoundError(
            f"No details_*.parquet files found in {baseline_results_dir} or "
            f"{candidate_results_dir}."
        )

    per_task: Dict[str, Dict[str, Any]] = {}
    grouped_baseline: Dict[str, List[Dict[str, Any]]] = {}
    grouped_candidate: Dict[str, List[Dict[str, Any]]] = {}
    all_baseline: List[Dict[str, Any]] = []
    all_candidate: List[Dict[str, Any]] = []
    for concrete in sorted(baseline):
        base_path, canonical, metric = baseline[concrete]
        candidate_path = candidate[concrete][0]
        base_rows = load_task_parquet_examples(base_path, metric)
        candidate_rows = load_task_parquet_examples(candidate_path, metric)
        stats = compute_paired_series_stats(
            base_rows,
            candidate_rows,
            concrete,
            num_resamples=num_resamples,
            seed=seed,
        )
        stats.update({"metric": metric, "canonical_benchmark": canonical})
        per_task[concrete] = stats
        grouped_baseline.setdefault(canonical, []).extend(base_rows)
        grouped_candidate.setdefault(canonical, []).extend(candidate_rows)
        all_baseline.extend(base_rows)
        all_candidate.extend(candidate_rows)

    per_canonical = {
        canonical: {
            **compute_paired_series_stats(
                grouped_baseline[canonical],
                grouped_candidate[canonical],
                canonical,
                num_resamples=num_resamples,
                seed=seed,
            ),
            "metric": TASK_SPEC_BY_KEY[canonical].metric,
        }
        for canonical in sorted(grouped_baseline)
    }
    return {
        "bootstrap_resamples": num_resamples,
        "bootstrap_seed": seed,
        "num_concrete_tasks_paired": len(per_task),
        "num_canonical_benchmarks_paired": len(per_canonical),
        "overall_pooled_stats": compute_paired_series_stats(
            all_baseline,
            all_candidate,
            "overall_pooled_examples",
            num_resamples=num_resamples,
            seed=seed,
        ),
        "per_canonical_benchmark": per_canonical,
        "per_concrete_task": per_task,
    }
