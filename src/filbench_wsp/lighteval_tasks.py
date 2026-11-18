"""Custom lighteval registry hook for FilBench split selection."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import filbench_evals as _upstream_filbench_evals
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    lighteval_dir = repo_root / "external" / "filbench-eval" / "lighteval"
    for p in [str(lighteval_dir), str(lighteval_dir / "community_tasks")]:
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)
    import filbench_evals as _upstream_filbench_evals

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig

from filbench_wsp.splits import (
    VALID_DATA_SPLITS,
    load_split_manifest,
    select_split_docs,
)

TATOEBA_PARQUET_REVISION = "7db98c55fbe29f031e67d043a2d460fb9888c196"


def build_patched_tasks_table(
    upstream_table: list[LightevalTaskConfig],
) -> list[LightevalTaskConfig]:
    """Clone upstream TASKS_TABLE and apply documented Hub schema/revision fixes."""
    patched: list[LightevalTaskConfig] = []
    for cfg in upstream_table:
        cfg_copy = copy.copy(cfg)
        if cfg_copy.hf_repo == "aisingapore/cultural_evaluation-kalahi":
            # Upstream Hub converted config 'default'/split 'tl' -> config 'tl'/split 'eval' (150 rows)
            cfg_copy.hf_subset = "tl"
            cfg_copy.hf_avail_splits = ["eval"]
            cfg_copy.evaluation_splits = ["eval"]
        elif cfg_copy.hf_repo == "Helsinki-NLP/tatoeba_mt":
            # Pin pre-wipe Parquet revision of v2021-08-07 where eng-tgl has 2,499 test rows (not 1 row)
            cfg_copy.hf_revision = TATOEBA_PARQUET_REVISION
        patched.append(cfg_copy)
    return patched


TASKS_TABLE = build_patched_tasks_table(list(_upstream_filbench_evals.TASKS_TABLE))
DATA_SPLIT_ENV = "FILBENCH_WSP_DATA_SPLIT"
SPLIT_MANIFEST_ENV = "FILBENCH_WSP_SPLIT_MANIFEST"


def install_lighteval_split_hook() -> None:
    if getattr(LightevalTask.eval_docs, "_filbench_wsp_wrapped", False):
        return

    original_eval_docs = LightevalTask.eval_docs

    def split_aware_eval_docs(self: LightevalTask) -> list[Any]:
        if self._docs is None:
            full_docs = list(original_eval_docs(self))
            raw_split = os.environ.get(DATA_SPLIT_ENV)
            if raw_split is None or not raw_split.strip():
                raise RuntimeError(
                    f"Missing required environment variable {DATA_SPLIT_ENV} "
                    "in lighteval subprocess."
                )
            active_split = raw_split.strip()
            if active_split not in VALID_DATA_SPLITS:
                raise ValueError(
                    f"Unsupported {DATA_SPLIT_ENV}={active_split!r}; "
                    f"expected one of {VALID_DATA_SPLITS}."
                )

            manifest_path = os.environ.get(SPLIT_MANIFEST_ENV, "").strip()
            manifest: Optional[dict[str, Any]] = None
            if manifest_path:
                manifest = load_split_manifest(Path(manifest_path))
            elif active_split in ("dev", "test_sealed"):
                raise RuntimeError(
                    f"{SPLIT_MANIFEST_ENV} must be set when "
                    f"{DATA_SPLIT_ENV}={active_split!r}."
                )

            self._docs = select_split_docs(
                self.name,
                full_docs,
                active_split,
                manifest,
            )
            print(
                f"[FilBench-WSP SplitHook] task={self.name} split={active_split} "
                f"selected={len(self._docs)}/{len(full_docs)}",
                flush=True,
            )
        return self._docs

    setattr(split_aware_eval_docs, "_filbench_wsp_wrapped", True)
    LightevalTask.eval_docs = split_aware_eval_docs  # type: ignore[method-assign]


install_lighteval_split_hook()


def _install_exit_guard() -> None:
    if os.name == "nt" and any("lighteval" in a for a in sys.argv):
        import atexit

        def _fast_exit():
            os._exit(0)

        atexit.register(_fast_exit)


_install_exit_guard()

