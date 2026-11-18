"""Unit tests for evaluation command construction and safety guards."""

import importlib.util
from pathlib import Path

import pytest


def _load_run_eval_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "03_run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_cli_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_mock_ckpt(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights_v1")
    return path


def test_dry_run_creates_manifest_and_blocks_overwrite(tmp_path: Path):
    module = _load_run_eval_module()
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = _make_mock_ckpt(tmp_path / "checkpoints" / "base" / "sailor2")
    experiments = tmp_path / "experiments"
    args = [
        "--exp-id", "smoke_001",
        "--condition", "baseline",
        "--base-checkpoint-dir", str(checkpoint),
        "--eval-checkpoint-dir", str(checkpoint),
        "--base-model-name", "sail/Sailor2-1B",
        "--eval-config", str(repo_root / "configs" / "eval_frozen.yaml"),
        "--tasks", "filbench|kalahi_tgl_mcf|0|0",
        "--experiments-root", str(experiments),
        "--dry-run",
    ]
    manifest = module.run_eval_cli(args)
    assert manifest.base_checkpoint_sha256 == manifest.output_checkpoint_sha256
    assert (experiments / "smoke_001" / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.run_eval_cli(args)
    module.run_eval_cli(args + ["--force"])


def test_edited_condition_cannot_point_at_base_checkpoint(tmp_path: Path):
    module = _load_run_eval_module()
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = _make_mock_ckpt(tmp_path / "checkpoints" / "base" / "sailor2")
    args = [
        "--exp-id", "bad_edit_001",
        "--condition", "llm_edited",
        "--base-checkpoint-dir", str(checkpoint),
        "--eval-checkpoint-dir", str(checkpoint),
        "--base-model-name", "sail/Sailor2-1B",
        "--eval-config", str(repo_root / "configs" / "eval_frozen.yaml"),
        "--tasks", "filbench|kalahi_tgl_mcf|0|0",
        "--experiments-root", str(tmp_path / "experiments"),
        "--dry-run",
    ]
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION"):
        module.run_eval_cli(args)
