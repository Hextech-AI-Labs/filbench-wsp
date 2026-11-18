"""Pytest collection guards for optional upstream evaluation dependencies.

The weight-editing safety tests must always run. Modules that require the
upstream FilBench/lighteval stack or Parquet support are skipped LOUDLY when
those dependencies are absent, so an environment gap can never masquerade as a
passing suite.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LE_DIR = _REPO / "external" / "filbench-eval" / "lighteval"

# Mirror the sys.path entries that filbench_wsp.lighteval_tasks and
# scripts/03_run_eval.py use at runtime, so the availability check below
# reflects how the code actually resolves the upstream checkout.
for _p in (_LE_DIR, _LE_DIR / "community_tasks"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

collect_ignore: list[str] = []
_missing: list[tuple[str, str]] = []


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


if not _available("pyarrow"):
    collect_ignore.append("test_paired_stats.py")
    _missing.append(("test_paired_stats.py", "pyarrow not installed  ->  python -m pip install pyarrow"))

if not _available("lighteval"):
    collect_ignore.append("test_lighteval_tasks.py")
    _missing.append(("test_lighteval_tasks.py", "lighteval not importable (upstream eval stack not installed)"))

if _missing:
    print("\n" + "=" * 78, file=sys.stderr)
    print("FILBENCH-WSP: upstream-dependent test modules NOT collected", file=sys.stderr)
    for _name, _reason in _missing:
        print(f"  - {_name}: {_reason}", file=sys.stderr)
    print("", file=sys.stderr)
    print("These are ENVIRONMENT GAPS, not passes. The weight-editing safety", file=sys.stderr)
    print("tests still ran. Install the deps and re-run the full suite before", file=sys.stderr)
    print("the evaluation phase.", file=sys.stderr)
    print("=" * 78 + "\n", file=sys.stderr)