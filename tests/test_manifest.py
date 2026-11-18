"""Unit tests for checkpoint hashing and experiment manifest safety controls."""

from pathlib import Path

import pytest

from filbench_wsp.manifest import (
    EditSummary,
    ExperimentManifest,
    compute_checkpoint_dir_hash,
)


def test_checkpoint_dir_hash_detects_single_byte_change(tmp_path: Path):
    checkpoint_dir = tmp_path / "mock_ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text('{"hidden_size": 1024}', encoding="utf-8")
    (checkpoint_dir / "model.safetensors").write_bytes(b"\x00\x01\x02\x03")

    hash_before = compute_checkpoint_dir_hash(checkpoint_dir)
    (checkpoint_dir / "model.safetensors").write_bytes(b"\x00\x01\x02\x04")

    assert hash_before != compute_checkpoint_dir_hash(checkpoint_dir)


def test_manifest_blocks_base_checkpoint_overwrite(tmp_path: Path):
    same_dir = str(tmp_path / "checkpoints" / "base" / "gemma")
    manifest = ExperimentManifest(
        experiment_id="exp_001_bad_edit",
        condition="llm_edited",
        base_checkpoint_path=same_dir,
        output_checkpoint_path=same_dir,
    )
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION"):
        manifest.validate_scientific_controls()


def test_manifest_roundtrip_save_and_load(tmp_path: Path):
    manifest_path = tmp_path / "experiments" / "exp_001" / "manifest.json"
    manifest = ExperimentManifest(
        experiment_id="exp_001_baseline",
        condition="baseline",
        base_model_name="google/gemma-3-1b-pt",
        base_checkpoint_sha256="abc123",
        output_checkpoint_sha256="abc123",
        edit_summary=EditSummary(total_params_edited_l0=0),
    )
    manifest.save(manifest_path)

    loaded = ExperimentManifest.load(manifest_path)
    assert loaded.experiment_id == "exp_001_baseline"
    assert loaded.base_model_name == "google/gemma-3-1b-pt"
    assert loaded.edit_summary.total_params_edited_l0 == 0
