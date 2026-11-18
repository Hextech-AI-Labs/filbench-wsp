"""Fail-closed evaluation runner wrapping the pinned FilBench lighteval checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from filbench_wsp.manifest import (
    ConditionType,
    EditSummary,
    ExperimentManifest,
    compute_checkpoint_dir_hash,
    compute_file_sha256,
    get_git_commit_sha,
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


def build_lighteval_model_args(checkpoint_dir: Path, eval_cfg: dict[str, Any]) -> str:
    """Build the deterministic model argument string for lighteval."""
    model_args = eval_cfg["model_args"]
    generation = eval_cfg["generation_parameters"]
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return (
        f"model_name={checkpoint_dir.resolve().as_posix()},"
        f"dtype={model_args['dtype']},"
        f"max_length={int(model_args['max_model_length'])},"
        f"device={device},batch_size=1,"
        f"generation_parameters={{temperature:{float(generation['temperature'])},"
        f"max_new_tokens:{int(generation['max_new_tokens'])},"
        f"seed:{int(generation['seed'])}}}"
    )


def build_lighteval_command(
    python_executable: str,
    backend: str,
    model_args_str: str,
    task_spec_str: str,
    custom_tasks_module: str,
    output_dir: Path,
    save_details: bool = True,
    use_chat_template: bool = False,
    max_samples: int | None = None,
    bootstrap_iters: int = 0,
) -> list[str]:
    """Construct one lighteval subprocess command."""
    command = [
        python_executable,
        "-m",
        "lighteval",
        backend,
        model_args_str,
        task_spec_str,
        "--custom-tasks",
        custom_tasks_module,
        "--output-dir",
        output_dir.resolve().as_posix(),
    ]
    if save_details:
        command.append("--save-details")
    if use_chat_template:
        command.append("--use-chat-template")
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    command.extend(["--bootstrap-iters", str(bootstrap_iters)])
    return command


def verify_lighteval_submodule_sha(lighteval_dir: Path, expected_sha_prefix: str) -> str:
    """Verify that the local lighteval checkout matches the frozen commit."""
    if not lighteval_dir.exists():
        raise FileNotFoundError(
            f"lighteval submodule directory not found at '{lighteval_dir}'."
        )
    actual_sha = get_git_commit_sha(lighteval_dir)
    if actual_sha == "uncommitted" or not actual_sha.startswith(expected_sha_prefix):
        raise RuntimeError(
            f"lighteval submodule commit mismatch in '{lighteval_dir}': "
            f"expected prefix '{expected_sha_prefix}', got '{actual_sha}'."
        )
    return actual_sha


def load_task_list_from_file(tasks_file: Path) -> list[str]:
    """Read non-empty, non-comment task specifications."""
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    tasks = [
        line.strip()
        for line in tasks_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tasks:
        raise ValueError(f"Tasks file '{tasks_file}' contains 0 task entries.")
    return tasks


def relocate_lighteval_outputs(checkpoint_dir: Path, results_dir: Path) -> None:
    """Contain outputs from the pinned lighteval checkout if it ignores output_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for output in checkpoint_dir.glob("results_*.json"):
        shutil.move(str(output), str(results_dir / output.name))
    for output in checkpoint_dir.iterdir():
        if output.is_dir() and "T" in output.name and output.name[:4].isdigit():
            shutil.move(str(output), str(results_dir / output.name))


def run_eval_cli(argv: list[str] | None = None) -> ExperimentManifest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--condition", required=True, choices=["baseline", "llm_edited", "random_control"])
    parser.add_argument("--base-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval_frozen.yaml"))
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/dev_split_manifest.json"))
    parser.add_argument(
        "--data-split",
        choices=["dev", "test_sealed", "full", "smoke_test"],
        default="dev",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--tasks-file", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--backend", choices=["accelerate", "vllm"])
    parser.add_argument("--lighteval-dir", type=Path, default=Path("external/filbench-eval/lighteval"))
    parser.add_argument("--lighteval-python", default=sys.executable)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.data_split in ("dev", "test_sealed") and args.max_samples is not None:
        raise ValueError("--max-samples cannot be combined with dev or test_sealed splits.")
    if args.data_split in ("dev", "test_sealed") and not args.split_manifest.exists():
        raise FileNotFoundError(
            f"Split manifest is required for {args.data_split}: {args.split_manifest}"
        )

    exp_dir = args.experiments_root / args.exp_id
    manifest_path = exp_dir / "manifest.json"
    results_dir = exp_dir / "results"
    if (manifest_path.exists() or results_dir.exists()) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory '{exp_dir}'. "
            "Use a new --exp-id or pass --force."
        )
    if args.force and results_dir.exists():
        shutil.rmtree(results_dir)

    eval_cfg = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    expected_sha = str(eval_cfg["filbench_lighteval_commit"])
    backend = args.backend or str(eval_cfg["backend"])
    base_hash = compute_checkpoint_dir_hash(args.base_checkpoint_dir)
    eval_hash = compute_checkpoint_dir_hash(args.eval_checkpoint_dir)
    condition: ConditionType = args.condition
    manifest = ExperimentManifest(
        experiment_id=args.exp_id,
        condition=condition,
        base_model_name=args.base_model_name,
        base_checkpoint_path=str(args.base_checkpoint_dir.resolve()),
        base_checkpoint_sha256=base_hash,
        git_commit_sha=get_git_commit_sha(),
        precision_dtype=str(eval_cfg["model_args"]["dtype"]),
        filbench_lighteval_sha=expected_sha,
        eval_backend=backend,
        eval_config_path=str(args.eval_config),
        data_split_used=args.data_split,
        split_manifest_sha256=compute_file_sha256(args.split_manifest)
        if args.split_manifest.exists()
        else "",
        random_seed=int(eval_cfg["generation_parameters"]["seed"]),
        edit_summary=EditSummary(
            total_params_edited_l0=0 if condition == "baseline" else -1
        ),
        output_checkpoint_path=str(args.eval_checkpoint_dir.resolve()),
        output_checkpoint_sha256=eval_hash,
    )
    manifest.validate_scientific_controls()

    if args.tasks:
        task_list = list(args.tasks)
    else:
        task_file = args.tasks_file or Path("configs/filbench_21_scored_tasks.txt")
        task_list = load_task_list_from_file(task_file)

    model_args = build_lighteval_model_args(args.eval_checkpoint_dir, eval_cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)
    if args.dry_run:
        command = build_lighteval_command(
            args.lighteval_python,
            backend,
            model_args,
            task_list[0],
            str(eval_cfg["custom_tasks_module"]),
            results_dir,
            bool(eval_cfg["runtime_flags"].get("save_details", True)),
            bool(eval_cfg["model_args"].get("use_chat_template", False)),
            args.max_samples,
            int(eval_cfg["runtime_flags"].get("bootstrap_iters", 0)),
        )
        print("[DRY RUN] Verified checkpoints and saved pre-run manifest.")
        print("  " + " ".join(command))
        return manifest

    ensure_hf_token()
    verify_lighteval_submodule_sha(args.lighteval_dir, expected_sha)
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath_entries = [
        str(repo_root / "src"),
        str(args.lighteval_dir.resolve()),
        str((args.lighteval_dir / "community_tasks").resolve()),
    ]
    proc_env = os.environ.copy()
    proc_env["FILBENCH_WSP_DATA_SPLIT"] = args.data_split
    proc_env["FILBENCH_WSP_SPLIT_MANIFEST"] = str(args.split_manifest.resolve())
    if os.environ.get("HF_TOKEN"):
        proc_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    existing_pythonpath = proc_env.get("PYTHONPATH")
    proc_env["PYTHONPATH"] = os.pathsep.join(
        pythonpath_entries + ([existing_pythonpath] if existing_pythonpath else [])
    )
    child_env = proc_env
    for task in task_list:
        subprocess.run(
            build_lighteval_command(
                args.lighteval_python,
                backend,
                model_args,
                task,
                str(eval_cfg["custom_tasks_module"]),
                results_dir,
                bool(eval_cfg["runtime_flags"].get("save_details", True)),
                bool(eval_cfg["model_args"].get("use_chat_template", False)),
                args.max_samples,
                int(eval_cfg["runtime_flags"].get("bootstrap_iters", 0)),
            ),
            cwd=args.lighteval_dir,
            env=proc_env,
            check=True,
        )
        relocate_lighteval_outputs(args.eval_checkpoint_dir, results_dir)

    if compute_checkpoint_dir_hash(args.base_checkpoint_dir) != base_hash:
        raise RuntimeError("CRITICAL TAMPER VIOLATION: Base checkpoint hash changed during evaluation!")
    if compute_checkpoint_dir_hash(args.eval_checkpoint_dir) != eval_hash:
        raise RuntimeError("CRITICAL TAMPER VIOLATION: Evaluated checkpoint hash changed during evaluation!")
    return manifest


if __name__ == "__main__":
    run_eval_cli()
