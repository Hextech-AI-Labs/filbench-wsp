from __future__ import annotations

import pytest

from filbench_wsp.splits import (
    build_task_split_manifest_entry,
    compute_dev_count,
    normalize_concrete_task_name,
    partition_task_indices,
    select_split_docs,
)


def test_partition_is_deterministic_disjoint_and_complete() -> None:
    dev, sealed = partition_task_indices("demo_task", 101)

    assert dev == partition_task_indices("demo_task", 101)[0]
    assert set(dev).isdisjoint(sealed)
    assert sorted(dev + sealed) == list(range(101))
    assert len(dev) == compute_dev_count(101)


def test_group_quota_is_capped_per_concrete_subtask() -> None:
    assert compute_dev_count(4015, 5) == 20
    assert compute_dev_count(100, 57) == 2
    assert compute_dev_count(2) == 1


def test_lighteval_task_names_are_normalized() -> None:
    assert normalize_concrete_task_name("filbench|global_mmlu_all_tgl_mcf:abstract_algebra|0|0") == (
        "global_mmlu_all_tgl_mcf:abstract_algebra"
    )
    assert partition_task_indices("filbench|global_mmlu_all_tgl_mcf:abstract_algebra", 20) == (
        partition_task_indices("global_mmlu_all_tgl_mcf:abstract_algebra", 20)
    )


def test_manifest_mismatch_is_rejected() -> None:
    entry = build_task_split_manifest_entry("demo_task", 20)
    entry["n_total"] = 21

    with pytest.raises(ValueError, match="n_total mismatch"):
        select_split_docs("demo_task", list(range(20)), "dev", {"concrete_tasks": {"demo_task": entry}})


def test_full_and_smoke_test_are_passthrough() -> None:
    docs = ["a", "b", "c"]

    assert select_split_docs("demo_task", docs, "full") == docs
    assert select_split_docs("demo_task", docs, "smoke_test") == docs


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid data_split"):
        select_split_docs("demo_task", [1, 2], "unknown")
