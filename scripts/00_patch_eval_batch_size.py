"""One-time patch: make the lighteval batch size configurable.

Locates build_lighteval_model_args (normally in scripts/03_run_eval.py), replaces
the whole function so it reads model_args.batch_size from configs/eval_frozen.yaml
instead of hardcoding batch_size=1. Writes a .bak backup and verifies the result
compiles. Idempotent; safe to re-run.

Usage (repo root):
  python scripts/00_patch_eval_batch_size.py --dry-run
  python scripts/00_patch_eval_batch_size.py
  python scripts/00_patch_eval_batch_size.py --file scripts/03_run_eval.py
"""
from __future__ import annotations

import argparse
import py_compile
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FUNC_NAME = "build_lighteval_model_args"
START_RE = re.compile(rf"^def {FUNC_NAME}\(", flags=re.MULTILINE)
END_RE = re.compile(r"^(?:def |class |@|if __name__)", flags=re.MULTILINE)

PREFERRED = [
    REPO / "scripts" / "03_run_eval.py",
    REPO / "src" / "filbench_wsp" / "eval_adapter.py",
]

NEW_FUNC = '''def build_lighteval_model_args(checkpoint_dir: Path, eval_cfg: dict[str, Any]) -> str:
    """Build the deterministic model argument string for lighteval."""
    model_args = eval_cfg["model_args"]
    generation = eval_cfg["generation_parameters"]
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    batch_size = int(model_args.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("model_args.batch_size must be >= 1")
    return (
        f"model_name={checkpoint_dir.resolve().as_posix()},"
        f"dtype={model_args['dtype']},"
        f"max_length={int(model_args['max_model_length'])},"
        f"device={device},batch_size={batch_size},"
        f"generation_parameters={{temperature:{float(generation['temperature'])},"
        f"max_new_tokens:{int(generation['max_new_tokens'])},"
        f"seed:{int(generation['seed'])}}}"
    )
'''


def is_patched(text: str) -> bool:
    return "batch_size={batch_size}" in text and bool(
        re.search(r"model_args\.get\(\s*['\"]batch_size['\"]", text)
    )


def candidate_files() -> list[Path]:
    seen: list[Path] = []
    for p in PREFERRED:
        if p.is_file() and p not in seen:
            seen.append(p)
    for d in ("scripts", "src"):
        root = REPO / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in str(p) or p.suffix == ".bak" or p in seen:
                continue
            seen.append(p)
    return seen


def locate() -> tuple[Path, str] | None:
    for p in candidate_files():
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if START_RE.search(text):
            return p, text
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Show what would change, write nothing")
    ap.add_argument("--file", type=Path, default=None, help="Force the file to patch (repo-relative)")
    args = ap.parse_args()

    if args.file is not None:
        target = (REPO / args.file).resolve() if not args.file.is_absolute() else args.file
        if not target.is_file():
            raise FileNotFoundError(f"Not found: {target}")
        text = target.read_text(encoding="utf-8")
        if not START_RE.search(text):
            raise RuntimeError(f"{target} does not define {FUNC_NAME} at column 0.")
    else:
        found = locate()
        if found is None:
            print(f"[ERROR] Could not find 'def {FUNC_NAME}(' in any file under scripts/ or src/.")
            print("        Run this to see where it lives:")
            print(r'        Select-String -Path scripts\*.py,src\filbench_wsp\*.py -Pattern "def '
                  + FUNC_NAME + '"')
            return 1
        target, text = found

    rel = target.relative_to(REPO).as_posix() if target.is_relative_to(REPO) else str(target)
    print(f"Located {FUNC_NAME} in: {rel}")

    if is_patched(text):
        print("[SKIP] Already patched. Nothing to do.")
        return 0

    m = START_RE.search(text)
    nxt = END_RE.search(text, m.end())
    end = nxt.start() if nxt else len(text)
    old_func = text[m.start():end]
    new_text = text[:m.start()] + NEW_FUNC + "\n\n" + text[end:]

    compile(new_text, str(target), "exec")  # fail before writing anything

    print(f"Replacing {len(old_func.splitlines())} lines -> {len(NEW_FUNC.splitlines())} lines")
    print(f"  old hardcodes 'batch_size=1,': {'batch_size=1,' in old_func}")
    print(f"  new hardcodes 'batch_size=1,': {'batch_size=1,' in NEW_FUNC}")

    if "from typing import" not in text or "Any" not in text:
        print("  [WARN] 'Any' may not be imported in this file; check the traceback if compile fails.")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return 0

    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    target.write_text(new_text, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    print(f"\n[OK] Patched: {rel}")
    print(f"     Backup:  {backup.relative_to(REPO).as_posix() if backup.is_relative_to(REPO) else backup}")
    print("\nREMINDER: batching changes padding counters, so re-run the baseline once")
    print("before comparing conditions (the parity check requires it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())