"""Auto-resolve missing third-party modules in the lighteval/FilBench import chain.

Repeatedly imports filbench_wsp.lighteval_tasks in a fresh interpreter; when the
fatal error is a ModuleNotFoundError for a third-party package, installs it with
uv pip into the CURRENT interpreter's environment and retries. Stops at the first
non-ModuleNotFoundError failure, a failed install, or when the chain imports clean.

Usage (with .venv-eval active):
  python scripts/00_fix_import_chain.py
"""
from __future__ import annotations

import re
import subprocess
import sys

TARGET_IMPORT = (
    "import filbench_wsp.lighteval_tasks as t; "
    "print('OK, tasks registered:', len(t.TASKS_TABLE))"
)
MAX_ROUNDS = 12
IGNORE = {"filbench_evals"}  # expected handled fallback inside lighteval_tasks.py
PIP_NAME = {  # module name -> pip name, where they differ
    "language_data": "language-data",
    "rouge_score": "rouge-score",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "cv2": "opencv-python",
}
MISSING_RE = re.compile(r"No module named '([^']+)'")


def run_probe() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", TARGET_IMPORT], capture_output=True, text=True)


def main() -> int:
    installed: list[str] = []
    for round_no in range(1, MAX_ROUNDS + 1):
        proc = run_probe()
        if proc.returncode == 0:
            print(proc.stdout.strip())
            print(f"\n[OK] Import chain clean. Installed along the way: {installed or 'nothing'}")
            return 0
        missing = [m for m in MISSING_RE.findall(proc.stderr) if m.split(".")[0] not in IGNORE]
        if not missing:
            print("[STOP] Probe failed with a non-missing-module error; manual fix needed:\n")
            print(proc.stderr)
            return 1
        module = missing[-1].split(".")[0]  # the FATAL one is last in the chained traceback
        pip_name = PIP_NAME.get(module, module)
        print(f"[round {round_no}] missing: {module} -> uv pip install {pip_name}")
        inst = subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, pip_name],
            capture_output=True,
            text=True,
        )
        if inst.returncode != 0:
            print(f"[STOP] install of {pip_name} failed:\n{inst.stdout}\n{inst.stderr}")
            return 1
        installed.append(pip_name)
    print(f"[STOP] Gave up after {MAX_ROUNDS} rounds; still failing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())