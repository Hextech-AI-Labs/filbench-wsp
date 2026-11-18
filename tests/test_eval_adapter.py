"""Tests for fail-closed FilBench scoring and input parity."""

import json
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from filbench_wsp.eval_adapter import (
    FILBENCH_SCORED_TASKS,
    compute_filbench_scores,
    score_results_directory,
    verify_cross_condition_input_parity,
)
from filbench_wsp.manifest import EditSummary, ExperimentManifest


def _load_cli_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "04_score_local.py"
    spec = importlib.util.spec_from_file_location("score_local_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(prompt: str = "prompt") -> dict[str, Any]:
    return {
        "hashes": {
            "hash_examples": "examples",
            "hash_full_prompts": prompt,
            "hash_input_tokens": "input",
            "hash_cont_tokens": "continuation",
        },
        "truncated": 0,
        "non_truncated": 10,
        "non_padded": 10,
        "padded": 0,
        "effective_few_shots": 0.0,
    }


def _config_tasks(*task_names: str) -> dict[str, Any]:
    return {
        f"filbench|{task_name}": {
            "name": task_name,
            "original_num_docs": 100,
            "effective_num_docs": 100,
            "hf_revision": None,
        }
        for task_name in task_names
    }


def test_canonical_task_count_and_instance_count():
    assert len(FILBENCH_SCORED_TASKS) == 21
    assert sum(task.canonical_n for task in FILBENCH_SCORED_TASKS) == 162510


def test_full_scores_include_macro_and_instance_weighted_results():
    metrics = {
        task.benchmark_key: 0.8 if task.category == "Reading Comprehension" else 0.4
        for task in FILBENCH_SCORED_TASKS
    }
    report = compute_filbench_scores(metrics)
    assert report.filbench_macro_score == 50.0
    assert report.filbench_instance_weighted_score > 62.0
    assert not report.is_partial_eval


def test_complete_dev_split_is_marked_subsampled_without_max_samples():
    metrics = {task.benchmark_key: 0.5 for task in FILBENCH_SCORED_TASKS}
    counts = {
        task.benchmark_key: {"effective": 1, "original": 1}
        for task in FILBENCH_SCORED_TASKS
    }

    report = compute_filbench_scores(
        metrics,
        allow_partial=True,
        doc_counts_by_task=counts,
    )

    assert report.is_subsampled
    assert report.is_partial_eval
    assert set(report.subsampled_tasks) == {
        key for key in metrics if not key.endswith(":_average")
    }


def test_missing_tasks_require_explicit_partial_mode():
    with pytest.raises(ValueError, match="Refusing to compute full FilBench score"):
        compute_filbench_scores({"kalahi_tgl_mcf": 0.6})

    report = compute_filbench_scores({"kalahi_tgl_mcf": 0.6}, allow_partial=True)
    assert report.is_partial_eval
    assert report.filbench_macro_score is None
    assert report.filbench_instance_weighted_score is None
    assert report.partial_macro_score_debug == 60.0


def test_scoring_modes_are_mutually_exclusive_and_recorded(tmp_path: Path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_filbench_scores(
            {"kalahi_tgl_mcf": 0.6},
            allow_partial=True,
            strict_missing_as_zero=True,
        )
    report = compute_filbench_scores({"kalahi_tgl_mcf": 0.6}, allow_partial=True)
    assert report.scoring_mode == "partial"
    zero_report = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.6}, strict_missing_as_zero=True
    )
    assert zero_report.scoring_mode == "missing_as_zero"


def test_unrecognized_metric_key_is_loud(tmp_path: Path):
    payload = {
        "config_general": {"model_name": "sail/Sailor2-1B"},
        "results": {"filbench|kalahi_tgl_mcf|0": {"wrong_metric": 0.75}},
        "summary_tasks": {"filbench|kalahi_tgl_mcf|0": _summary()},
    }
    (tmp_path / "results_01.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError, match="expected metric 'acc' was not found"):
        score_results_directory(tmp_path, allow_partial=True)


def test_flat_summary_hashes_are_rejected(tmp_path: Path):
    payload = {
        "config_general": {"model_name": "sail/Sailor2-1B"},
        "results": {"filbench|kalahi_tgl_mcf|0": {"acc": 0.6}},
        "summary_tasks": {"filbench|kalahi_tgl_mcf|0": {"hash_examples": "flat"}},
    }
    (tmp_path / "results_01.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing the nested 'hashes' dict"):
        score_results_directory(tmp_path, allow_partial=True)


def test_duplicate_tasks_and_mixed_models_are_rejected(tmp_path: Path):
    def write(name: str, task: str, model: str, score: float) -> None:
        payload = {
            "config_general": {"model_name": model},
            "results": {f"filbench|{task}|0": {"acc": score}},
            "config_tasks": _config_tasks(task),
            "summary_tasks": {f"filbench|{task}|0": _summary()},
        }
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    write("results_01.json", "kalahi_tgl_mcf", "base", 0.6)
    write("results_02.json", "kalahi_tgl_mcf", "base", 0.7)
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        score_results_directory(tmp_path, allow_partial=True)

    (tmp_path / "results_02.json").unlink()
    write("results_02.json", "cebuaner_ceb_mcf", "edited", 0.7)
    with pytest.raises(ValueError, match="Multiple model_name"):
        score_results_directory(tmp_path, allow_partial=True)


def test_parity_detects_hash_drift_vanished_tasks_and_missing_hashes():
    base = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.5, "cebuaner_ceb_mcf": 0.6},
        allow_partial=True,
        task_summaries={
            "kalahi_tgl_mcf": _summary("a"),
            "cebuaner_ceb_mcf": _summary("b"),
        },
    )
    matching = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.55, "cebuaner_ceb_mcf": 0.65},
        allow_partial=True,
        task_summaries={
            "kalahi_tgl_mcf": _summary("a"),
            "cebuaner_ceb_mcf": _summary("b"),
        },
    )
    assert verify_cross_condition_input_parity(base, matching) == []

    drifted = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.55, "cebuaner_ceb_mcf": 0.65},
        allow_partial=True,
        task_summaries={
            "kalahi_tgl_mcf": _summary("drift"),
            "cebuaner_ceb_mcf": _summary("b"),
        },
    )
    assert any("hashes['hash_full_prompts']" in error for error in verify_cross_condition_input_parity(base, drifted))

    vanished = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.55},
        allow_partial=True,
        task_summaries={"kalahi_tgl_mcf": _summary("a")},
    )
    assert any("vanished in comparison run" in error for error in verify_cross_condition_input_parity(base, vanished))

    corrupted = compute_filbench_scores(
        {"kalahi_tgl_mcf": 0.55, "cebuaner_ceb_mcf": 0.65},
        allow_partial=True,
        task_summaries={
            "kalahi_tgl_mcf": {"hash_examples": "flat"},
            "cebuaner_ceb_mcf": _summary("b"),
        },
    )
    assert any("missing nested 'hashes' dict" in error for error in verify_cross_condition_input_parity(base, corrupted))


def test_missing_as_zero_parity_uses_actually_ran_tasks(tmp_path: Path):
    base_dir = tmp_path / "base"
    comparison_dir = tmp_path / "comparison"
    base_dir.mkdir()
    comparison_dir.mkdir()
    base_payload = {
        "config_general": {"model_name": "base"},
        "results": {
            "filbench|kalahi_tgl_mcf|0": {"acc": 0.6},
            "filbench|cebuaner_ceb_mcf|0": {"acc": 0.7},
        },
        "summary_tasks": {
            "filbench|kalahi_tgl_mcf|0": _summary(),
            "filbench|cebuaner_ceb_mcf|0": _summary(),
        },
        "config_tasks": _config_tasks("kalahi_tgl_mcf", "cebuaner_ceb_mcf"),
    }
    comparison_payload = {
        "config_general": {"model_name": "base"},
        "results": {"filbench|kalahi_tgl_mcf|0": {"acc": 0.6}},
        "summary_tasks": {"filbench|kalahi_tgl_mcf|0": _summary()},
        "config_tasks": _config_tasks("kalahi_tgl_mcf"),
    }
    (base_dir / "results_01.json").write_text(json.dumps(base_payload), encoding="utf-8")
    (comparison_dir / "results_01.json").write_text(
        json.dumps(comparison_payload), encoding="utf-8"
    )
    base = score_results_directory(base_dir, strict_missing_as_zero=True)
    comparison = score_results_directory(comparison_dir, strict_missing_as_zero=True)
    assert len(base.per_task_scores_pct) == 21
    errors = verify_cross_condition_input_parity(base, comparison)
    assert any("vanished in comparison run: ['cebuaner_ceb_mcf']" in error for error in errors)


def test_missing_model_name_is_rejected(tmp_path: Path):
    payload = {
        "config_general": {},
        "results": {"filbench|kalahi_tgl_mcf|0": {"acc": 0.6}},
        "summary_tasks": {"filbench|kalahi_tgl_mcf|0": _summary()},
        "config_tasks": _config_tasks("kalahi_tgl_mcf"),
    }
    (tmp_path / "results_01.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model_name.*missing or empty"):
        score_results_directory(tmp_path, allow_partial=True)


def test_cli_refuses_overwrite_without_force(tmp_path: Path):
    cli = _load_cli_module()
    experiment_dir = tmp_path / "experiment"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True)
    payload = {
        "config_general": {"model_name": "base"},
        "results": {"filbench|kalahi_tgl_mcf|0": {"acc": 0.6}},
        "summary_tasks": {"filbench|kalahi_tgl_mcf|0": _summary()},
        "config_tasks": _config_tasks("kalahi_tgl_mcf"),
    }
    (results_dir / "results_01.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = experiment_dir / "manifest.json"
    ExperimentManifest(
        experiment_id="experiment",
        condition="baseline",
        base_checkpoint_sha256="hash",
        output_checkpoint_sha256="hash",
        edit_summary=EditSummary(),
    ).save(manifest_path)

    cli.score_experiment_cli(["--exp-dir", str(experiment_dir), "--partial"])
    assert ExperimentManifest.load(manifest_path).scoring_mode == "partial"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.score_experiment_cli(["--exp-dir", str(experiment_dir), "--partial"])
    cli.score_experiment_cli(["--exp-dir", str(experiment_dir), "--partial", "--force"])


REAL_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real"


def _copy_real_fixture(tmp_path: Path, fixture_name: str, output_name: str) -> Path:
    output_dir = tmp_path / output_name
    output_dir.mkdir()
    (output_dir / "results_fixture.json").write_bytes(
        (REAL_FIXTURES_DIR / fixture_name).read_bytes()
    )
    return output_dir


def test_real_lighteval_null_parity_between_two_identical_checkpoint_runs(tmp_path: Path):
    run_a = _copy_real_fixture(
        tmp_path,
        "sailor2_mmlu_abstract_algebra_run_a.json",
        "run_a",
    )
    run_b = _copy_real_fixture(
        tmp_path,
        "sailor2_mmlu_abstract_algebra_max2.json",
        "run_b",
    )
    summary_a = score_results_directory(run_a, allow_partial=True)
    summary_b = score_results_directory(run_b, allow_partial=True)

    assert verify_cross_condition_input_parity(
        summary_a, summary_b, expect_identical_outputs=True
    ) == []
    assert summary_a.per_task_scores_pct == summary_b.per_task_scores_pct


def test_real_ntrex_fixture_ignores_generation_continuation_drift(tmp_path: Path):
    run_dir = _copy_real_fixture(
        tmp_path, "sailor2_ntrex128_fil_max2.json", "ntrex"
    )
    summary = score_results_directory(run_dir, allow_partial=True)
    assert summary.per_task_scores_pct["ntrex128_fil"] == pytest.approx(4.878)
    assert summary.doc_counts_by_task["ntrex128_fil"] == {
        "effective": 2,
        "original": 1997,
    }


def test_real_dengue_fixture_accepts_synthetic_average_row(tmp_path: Path):
    run_dir = _copy_real_fixture(
        tmp_path, "sailor2_dengue_filipino_fil_max2.json", "dengue"
    )
    summary = score_results_directory(run_dir, allow_partial=True)
    assert summary.per_task_scores_pct["dengue_filipino_fil:_average"] == pytest.approx(50.0)
    assert summary.doc_counts_by_task["dengue_filipino_fil:_average"] == {
        "effective": 10,
        "original": 20075,
    }
