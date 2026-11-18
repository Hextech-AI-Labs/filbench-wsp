"""Deterministic dev and sealed-test split partitioning."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence, TypeVar

T = TypeVar("T")

MASTER_SPLIT_SEED = 42
VALID_DATA_SPLITS = ("dev", "test_sealed", "full", "smoke_test")
GROUP_SUBTASK_COUNTS = {
    "dengue_filipino_fil": 5,
    "include_tgl_mcf": 4,
    "global_mmlu_all_tgl_mcf": 57,
}


def normalize_concrete_task_name(task_name: str) -> str:
    """Strip lighteval's suite prefix and task-spec suffixes."""
    parts = task_name.strip().split("|")
    if len(parts) >= 2 and parts[0] == "filbench":
        return parts[1]
    return parts[0]


def get_group_subtask_count(task_name: str) -> int:
    normalized = normalize_concrete_task_name(task_name)
    return GROUP_SUBTASK_COUNTS.get(normalized.split(":", 1)[0], 1)


def derive_task_seed(task_name: str, master_seed: int = MASTER_SPLIT_SEED) -> int:
    normalized = normalize_concrete_task_name(task_name)
    digest = hashlib.sha256(f"{master_seed}:{normalized}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_dev_count(n_total: int, num_subtasks_in_group: int = 1) -> int:
    if n_total <= 1:
        raise ValueError(f"Task must have at least 2 examples, got {n_total}.")
    if num_subtasks_in_group <= 1:
        target = min(max(20, int(0.10 * n_total)), 100)
    else:
        target = max(
            2,
            min((100 + num_subtasks_in_group - 1) // num_subtasks_in_group, int(0.10 * n_total)),
        )
    return max(1, min(target, n_total // 2))


def hash_index_list(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def partition_task_indices(
    task_name: str,
    n_total: int,
    master_seed: int = MASTER_SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    normalized = normalize_concrete_task_name(task_name)
    n_dev = compute_dev_count(n_total, get_group_subtask_count(normalized))
    indices = list(range(n_total))
    random.Random(derive_task_seed(normalized, master_seed)).shuffle(indices)
    return sorted(indices[:n_dev]), sorted(indices[n_dev:])


def build_task_split_manifest_entry(
    task_name: str,
    n_total: int,
    master_seed: int = MASTER_SPLIT_SEED,
) -> dict[str, Any]:
    dev_indices, sealed_indices = partition_task_indices(task_name, n_total, master_seed)
    return {
        "n_total": n_total,
        "n_dev": len(dev_indices),
        "n_test_sealed": len(sealed_indices),
        "num_subtasks_in_group": get_group_subtask_count(task_name),
        "task_seed": derive_task_seed(task_name, master_seed),
        "dev_indices": dev_indices,
        "dev_indices_sha256": hash_index_list(dev_indices),
        "sealed_test_indices_sha256": hash_index_list(sealed_indices),
    }


def select_split_docs(
    task_name: str,
    full_docs: Sequence[T],
    data_split: str,
    manifest_data: dict[str, Any] | None = None,
) -> list[T]:
    if data_split not in VALID_DATA_SPLITS:
        raise ValueError(f"Invalid data_split={data_split!r}; expected {VALID_DATA_SPLITS}.")
    if data_split in ("full", "smoke_test"):
        return list(full_docs)

    normalized = normalize_concrete_task_name(task_name)
    master_seed = (
        int(manifest_data.get("master_seed", MASTER_SPLIT_SEED))
        if isinstance(manifest_data, dict)
        else MASTER_SPLIT_SEED
    )
    dev_indices, sealed_indices = partition_task_indices(
        normalized, len(full_docs), master_seed
    )
    if isinstance(manifest_data, dict):
        tasks = manifest_data.get("concrete_tasks", manifest_data.get("tasks", {}))
        entry = tasks.get(normalized) if isinstance(tasks, dict) else None
        if isinstance(entry, dict):
            if entry.get("n_total") is not None and int(entry["n_total"]) != len(full_docs):
                raise ValueError(f"Split manifest n_total mismatch for '{normalized}'.")
            if entry.get("dev_indices") is not None and list(entry["dev_indices"]) != dev_indices:
                raise ValueError(f"Split manifest dev_indices mismatch for '{normalized}'.")
            actual_hash = hash_index_list(sealed_indices)
            if entry.get("sealed_test_indices_sha256") not in (None, actual_hash):
                raise ValueError(f"Split manifest sealed indices hash mismatch for '{normalized}'.")

    selected = dev_indices if data_split == "dev" else sealed_indices
    return [full_docs[index] for index in selected]


def load_split_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Malformed split manifest at {manifest_path}")
    return data
