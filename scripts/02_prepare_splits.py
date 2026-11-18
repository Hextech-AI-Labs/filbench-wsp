"""Generate the deterministic FilBench dev/sealed-test split manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from filbench_wsp.eval_adapter import FILBENCH_SCORED_TASKS
from filbench_wsp.splits import (
    MASTER_SPLIT_SEED,
    build_task_split_manifest_entry,
)


def ensure_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            import huggingface_hub

            token = huggingface_hub.get_token()
            if token:
                os.environ["HF_TOKEN"] = token
        except Exception:
            pass
    return token


KNOWN_CONCRETE_SUBTASK_COUNTS = {
    "dengue_filipino_fil:absent": 4015,
    "dengue_filipino_fil:dengue": 4015,
    "dengue_filipino_fil:health": 4015,
    "dengue_filipino_fil:mosquito": 4015,
    "dengue_filipino_fil:sick": 4015,
    "global_mmlu_all_tgl_mcf:abstract_algebra": 100,
}
GROUP_PREFIXES = {
    "dengue_filipino_fil",
    "global_mmlu_all_tgl_mcf",
    "include_tgl_mcf",
}


def discover_live_task_counts(lighteval_dir: Path) -> dict[str, int]:
    """Load every scored concrete task and return its observed eval-doc count."""
    ensure_hf_token()
    os.environ["FILBENCH_WSP_DATA_SPLIT"] = "full"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    repo_root = Path(__file__).resolve().parents[1]
    lighteval_root = lighteval_dir.resolve()
    sys.path[:0] = [
        str(repo_root / "src"),
        str(lighteval_root),
        str(lighteval_root / "community_tasks"),
    ]
    from filbench_wsp.lighteval_tasks import TASKS_TABLE
    from lighteval.tasks.lighteval_task import LightevalTask

    standalone = {
        spec.benchmark_key
        for spec in FILBENCH_SCORED_TASKS
        if not spec.benchmark_key.endswith(":_average")
        and spec.benchmark_key.split(":", 1)[0] not in GROUP_PREFIXES
    }
    configs = {
        config.name: config
        for config in TASKS_TABLE
        if config.name in standalone
        or config.name.split(":", 1)[0] in GROUP_PREFIXES
    }
    failures: list[str] = []
    counts: dict[str, int] = {}
    for task_name, config in sorted(configs.items()):
        max_retries = 3
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                count = len(LightevalTask(task_name, config).eval_docs())
                if count < 2:
                    raise ValueError(f"dataset produced only {count} evaluation docs")
                counts[task_name] = count
                print(f"[OK] {task_name}: n_total={count}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    import time

                    print(f"[RETRY {attempt}/{max_retries}] {task_name}: {exc} - retrying...")
                    time.sleep(2 * attempt)
        if last_exc is not None:
            failures.append(f"{task_name}: {type(last_exc).__name__}: {last_exc}")
            print(f"[FAIL] {failures[-1]}")
    if failures:
        raise RuntimeError(
            "Live dataset verification failed for:\n" + "\n".join(failures)
        )
    expected_count = len(standalone) + sum(
        1 for name in configs if name.split(":", 1)[0] in GROUP_PREFIXES
    )
    if len(counts) != expected_count:
        raise RuntimeError(
            f"Live dataset verification loaded {len(counts)} tasks; "
            f"expected {expected_count}."
        )
    return counts


def generate_split_manifest(master_seed: int = MASTER_SPLIT_SEED) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for spec in FILBENCH_SCORED_TASKS:
        if not spec.benchmark_key.endswith(":_average"):
            tasks[spec.benchmark_key] = build_task_split_manifest_entry(
                spec.benchmark_key, spec.canonical_n, master_seed
            )
    for task_name, n_total in KNOWN_CONCRETE_SUBTASK_COUNTS.items():
        tasks[task_name] = build_task_split_manifest_entry(
            task_name, n_total, master_seed
        )
    return {
        "schema_version": "1.1.0",
        "master_seed": master_seed,
        "split_rule": (
            "Standalone: min(min(max(20, floor(0.10*N)), 100), floor(N/2)); "
            "groups: min(max(2, min(ceil(100/K), floor(0.10*N))), floor(N/2))."
        ),
        "concrete_tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/dev_split_manifest.json"))
    parser.add_argument("--seed", type=int, default=MASTER_SPLIT_SEED)
    parser.add_argument(
        "--verify-live-datasets",
        action="store_true",
        help="Load all 84 concrete tasks and use their observed eval-doc counts.",
    )
    parser.add_argument(
        "--lighteval-dir",
        type=Path,
        default=Path("external/filbench-eval/lighteval"),
    )
    args = parser.parse_args()
    if args.verify_live_datasets:
        live_counts = discover_live_task_counts(args.lighteval_dir)
        payload = {
            "schema_version": "1.1.0",
            "master_seed": args.seed,
            "split_rule": (
                "Standalone: min(min(max(20, floor(0.10*N)), 100), floor(N/2)); "
                "groups: min(max(2, min(ceil(100/K), floor(0.10*N))), floor(N/2))."
            ),
            "concrete_tasks": {
                name: build_task_split_manifest_entry(name, count, args.seed)
                for name, count in sorted(live_counts.items())
            },
        }
    else:
        payload = generate_split_manifest(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Wrote split manifest with {len(payload['concrete_tasks'])} concrete tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
