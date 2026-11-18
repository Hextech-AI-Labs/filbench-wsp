"""Shared fixtures for verified lighteval JSON structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REAL_SMOKE_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "real"
    / "sailor2_mmlu_abstract_algebra_max2.json"
)


def load_real_smoke_fixture() -> dict[str, Any]:
    with REAL_SMOKE_FIXTURE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _make_assumed_summary(
    hash_examples: str = "ex_hash_001",
    hash_full_prompts: str = "pr_hash_001",
    hash_input_tokens: str = "in_tok_001",
    hash_cont_tokens: str = "ct_tok_001",
    non_truncated: int = 150,
    non_padded: int = 150,
) -> dict[str, Any]:
    return {
        "hashes": {
            "hash_examples": hash_examples,
            "hash_full_prompts": hash_full_prompts,
            "hash_input_tokens": hash_input_tokens,
            "hash_cont_tokens": hash_cont_tokens,
        },
        "truncated": 0,
        "non_truncated": non_truncated,
        "non_padded": non_padded,
        "padded": 0,
        "effective_few_shots": 0.0,
    }


def _make_assumed_results_payload(
    model_name: str | None = "sail/Sailor2-1B",
    max_samples: int | None = None,
    task_metrics: dict[str, dict[str, float]] | None = None,
    task_summaries: dict[str, dict[str, Any]] | None = None,
    config_tasks: dict[str, dict[str, Any]] | None = None,
    original_num_docs: int = 150,
    effective_num_docs: int = 150,
) -> dict[str, Any]:
    if task_metrics is None:
        task_metrics = {"filbench|kalahi_tgl_mcf|0": {"acc": 0.60}}
    if task_summaries is None:
        task_summaries = {
            full_key: _make_assumed_summary() for full_key in task_metrics
        }
    if config_tasks is None:
        config_tasks = {
            f"filbench|{full_key.split('|')[1]}": {
                "name": full_key.split("|")[1],
                "hf_repo": "mock/repo",
                "hf_subset": "default",
                "hf_revision": None,
                "original_num_docs": original_num_docs,
                "effective_num_docs": effective_num_docs,
            }
            for full_key in task_metrics
        }
    config_general = {"max_samples": max_samples, "model_sha": ""}
    if model_name is not None:
        config_general["model_name"] = model_name
    return {
        "config_general": config_general,
        "results": task_metrics,
        "config_tasks": config_tasks,
        "summary_tasks": task_summaries,
    }
