"""Fail-closed CLI scorer with explicit overwrite protection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filbench_wsp.eval_adapter import score_results_directory, verify_cross_condition_input_parity
from filbench_wsp.manifest import ExperimentManifest
from filbench_wsp.paired_stats import compare_experiment_details_dirs


def score_experiment_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute local FilBench scores for an experiment.")
    parser.add_argument("--exp-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--partial", action="store_true")
    modes.add_argument("--allow-missing-as-zero", action="store_true")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    score_path = args.exp_dir / "filbench_score.json"
    manifest_path = args.exp_dir / "manifest.json"
    if score_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing score file '{score_path}'. Pass --force to overwrite."
        )

    existing_manifest: ExperimentManifest | None = None
    if manifest_path.exists():
        existing_manifest = ExperimentManifest.load(manifest_path)
        already_scored = (
            bool(existing_manifest.scoring_mode)
            or existing_manifest.num_tasks_evaluated > 0
            or bool(existing_manifest.per_task_results)
        )
        if already_scored and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite already-scored manifest '{manifest_path}' "
                f"(scoring_mode='{existing_manifest.scoring_mode}'). Pass --force to overwrite."
            )

    report = score_results_directory(
        args.exp_dir / "results",
        allow_partial=args.partial,
        strict_missing_as_zero=args.allow_missing_as_zero,
    )
    paired_comparison = None
    if args.baseline_dir is not None:
        baseline = score_results_directory(
            args.baseline_dir / "results",
            allow_partial=args.partial,
            strict_missing_as_zero=args.allow_missing_as_zero,
        )
        mismatches = verify_cross_condition_input_parity(baseline, report)
        if mismatches:
            raise RuntimeError(
                "CRITICAL PARITY FAILURE between baseline and experiment:\n  - "
                + "\n  - ".join(mismatches)
            )
        print("Verified input parity against baseline.")
        baseline_details = list((args.baseline_dir / "results").rglob("details_*.parquet"))
        experiment_details = list((args.exp_dir / "results").rglob("details_*.parquet"))
        if baseline_details and experiment_details:
            paired_comparison = compare_experiment_details_dirs(
                args.baseline_dir / "results",
                args.exp_dir / "results",
            )
            paired_path = args.exp_dir / "paired_comparison.json"
            paired_path.write_text(
                json.dumps(paired_comparison, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Saved paired comparison report to: {paired_path}")

    score_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    score_summary = report.to_dict()
    if paired_comparison is not None:
        score_summary["paired_comparison"] = paired_comparison
    (args.exp_dir / "score_summary.json").write_text(
        json.dumps(score_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved FilBench score report to: {score_path}")
    print(f"  Scoring mode: {report.scoring_mode}")
    print(f"  Tasks evaluated: {report.num_tasks_evaluated} / {report.total_canonical_tasks}")

    if existing_manifest is not None:
        existing_manifest.scoring_mode = report.scoring_mode
        existing_manifest.is_partial_eval = report.is_partial_eval
        existing_manifest.is_subsampled = report.is_subsampled
        existing_manifest.max_samples_config = report.max_samples_config
        existing_manifest.num_tasks_evaluated = report.num_tasks_evaluated
        existing_manifest.evaluated_tasks = report.evaluated_tasks
        existing_manifest.missing_tasks = report.missing_tasks
        existing_manifest.subsampled_tasks = report.subsampled_tasks
        existing_manifest.doc_counts_by_task = report.doc_counts_by_task
        existing_manifest.dataset_revisions = report.dataset_revisions
        existing_manifest.task_source_files = report.source_files_by_task
        existing_manifest.per_task_results = report.per_task_scores_pct
        existing_manifest.subtask_results = report.subtask_scores_pct
        existing_manifest.category_scores = report.category_scores
        existing_manifest.filbench_macro_score = report.filbench_macro_score
        existing_manifest.filbench_instance_weighted_score = report.filbench_instance_weighted_score
        existing_manifest.partial_macro_score_debug = report.partial_macro_score_debug
        existing_manifest.save(manifest_path)
        print(f"Updated experiment manifest: {manifest_path}")


if __name__ == "__main__":
    score_experiment_cli()
