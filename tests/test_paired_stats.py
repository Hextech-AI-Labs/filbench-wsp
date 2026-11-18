from __future__ import annotations

from pathlib import Path

import pytest

from filbench_wsp.paired_stats import (
    compare_experiment_details_dirs,
    compute_paired_series_stats,
    resolve_concrete_task_from_parquet_name,
)


def test_resolves_standalone_and_sanitized_group_names() -> None:
    assert resolve_concrete_task_from_parquet_name(
        "details_filbench_ntrex128_fil_0_2026-01-01T00-00-00.parquet"
    ) == ("ntrex128_fil", "ntrex128_fil", "rougeL")
    assert resolve_concrete_task_from_parquet_name(
        "details_filbench_balita_tgl_mcf_0_2026-01-01T00-00-00.parquet"
    ) == ("balita_tgl_mcf", "balita_tgl_mcf", "acc")
    assert resolve_concrete_task_from_parquet_name(
        "details_filbench_dengue_filipino_fil_absent_0_2026-01-01T00-00-00.parquet"
    ) == ("dengue_filipino_fil:absent", "dengue_filipino_fil:_average", "acc_norm")
    assert resolve_concrete_task_from_parquet_name(
        "details_filbench_global_mmlu_all_tgl_mcf_abstract_algebra_0_2026-01-01T00-00-00.parquet"
    )[0] == "global_mmlu_all_tgl_mcf:abstract_algebra"
    assert resolve_concrete_task_from_parquet_name(
        "details_filbench_include_tgl_mcf_driving_license_0_2026-01-01T00-00-00.parquet"
    )[0] == "include_tgl_mcf:driving_license"


def test_rejects_unknown_detail_task() -> None:
    with pytest.raises(ValueError, match="does not map"):
        resolve_concrete_task_from_parquet_name(
            "details_filbench_unknown_task_0_2026-01-01T00-00-00.parquet"
        )


def test_computes_wins_losses_and_ties() -> None:
    base = [
        {"example": "a", "full_prompt": "p1", "score_pct": 0},
        {"example": "b", "full_prompt": "p2", "score_pct": 100},
        {"example": "c", "full_prompt": "p3", "score_pct": 50},
    ]
    candidate = [
        {"example": "a", "full_prompt": "p1", "score_pct": 100},
        {"example": "b", "full_prompt": "p2", "score_pct": 0},
        {"example": "c", "full_prompt": "p3", "score_pct": 50},
    ]

    stats = compute_paired_series_stats(base, candidate, "synthetic", num_resamples=100)

    assert stats["n_improved"] == 1
    assert stats["n_regressed"] == 1
    assert stats["n_unchanged"] == 1
    assert stats["mean_delta_pct"] == 0.0
    assert stats["sign_test_p_two_sided"] == 1.0


def test_alignment_mismatch_and_row_count_fail_closed() -> None:
    base = [{"example": "a", "full_prompt": "p1", "score_pct": 0}]
    candidate = [{"example": "a", "full_prompt": "tampered", "score_pct": 100}]

    with pytest.raises(ValueError, match="prompt alignment mismatch"):
        compute_paired_series_stats(base, candidate, "synthetic")
    with pytest.raises(ValueError, match="row count mismatch"):
        compute_paired_series_stats(base, [], "synthetic")


def test_real_null_parquet_pair_is_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = root / "experiments" / "smoke_null_a" / "results"
    candidate = root / "experiments" / "smoke_mcf_b" / "results"
    if not baseline.exists() or not candidate.exists():
        pytest.skip("Real smoke artifacts are unavailable.")

    result = compare_experiment_details_dirs(baseline, candidate)
    stats = result["overall_pooled_stats"]
    assert stats["mean_delta_pct"] == 0.0
    assert stats["n_improved"] == 0
    assert stats["n_regressed"] == 0
    assert stats["n_unchanged"] == 2
    assert stats["sign_test_p_two_sided"] == 1.0
