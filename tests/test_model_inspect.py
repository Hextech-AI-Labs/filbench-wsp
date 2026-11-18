"""Unit tests for hardware detection, memory estimation, and spectral profiling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from filbench_wsp.model_inspect import (
    BYTES_PER_DTYPE,
    compute_tensor_spectral_summary,
    get_hardware_status,
    inspect_checkpoint_spectral_profile,
)


def test_hardware_status_runs():
    status = get_hardware_status()
    assert isinstance(status.cuda_available, bool)
    assert isinstance(status.total_vram_gb, float)


def test_kv_cache_math_1b_gqa():
    """The bf16 GQA example should occupy exactly 0.125 GiB."""
    b = BYTES_PER_DTYPE["bfloat16"]
    kv_bytes = 2 * 1 * 4096 * 16 * 8 * 64 * b
    kv_gb = kv_bytes / (1024**3)
    assert kv_gb == 0.125


def test_compute_tensor_spectral_summary_1d():
    t = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.bfloat16)
    summary = compute_tensor_spectral_summary("norm.weight", t)

    assert summary.name == "norm.weight"
    assert summary.is_2d is False
    assert summary.shape == [4]
    assert summary.numel == 4
    # Frobenius norm = sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477225
    assert summary.frobenius_norm == pytest.approx(5.477225, rel=1e-3)
    assert summary.linf_norm == pytest.approx(4.0, rel=1e-3)
    assert summary.min_val == pytest.approx(-4.0, rel=1e-3)
    assert summary.max_val == pytest.approx(3.0, rel=1e-3)
    assert summary.spectral_norm is None
    assert summary.condition_number is None
    assert summary.effective_rank is None
    assert summary.band_energy_fractions is None


def test_compute_tensor_spectral_summary_2d():
    # Construct a 4x4 matrix with exact singular values: 10.0, 5.0, 2.0, 1.0
    diag_vals = torch.tensor([10.0, 5.0, 2.0, 1.0], dtype=torch.float32)
    t = torch.diag(diag_vals)
    summary = compute_tensor_spectral_summary("layer.weight", t)

    assert summary.name == "layer.weight"
    assert summary.is_2d is True
    assert summary.shape == [4, 4]
    assert summary.numel == 16
    assert summary.max_rank == 4
    assert summary.spectral_norm == pytest.approx(10.0, rel=1e-3)
    assert summary.min_singular_value == pytest.approx(1.0, rel=1e-3)
    assert summary.condition_number == pytest.approx(10.0, rel=1e-3)
    assert summary.top_5_singular_values == [10.0, 5.0, 2.0, 1.0]

    # Effective rank must be between 1.0 and 4.0
    assert summary.effective_rank is not None
    assert 1.0 <= summary.effective_rank <= 4.0

    # Total energy: 100 + 25 + 4 + 1 = 130
    # 50% energy = 65 -> achieved at rank 1 (100 >= 65)
    assert summary.energy_rank_50pct == 1
    # 80% energy = 104 -> achieved at rank 2 (125 >= 104)
    assert summary.energy_rank_80pct == 2
    # 95% energy = 123.5 -> achieved at rank 2 (125 >= 123.5)
    assert summary.energy_rank_95pct == 2

    assert summary.band_energy_fractions is not None
    q_sum = sum(summary.band_energy_fractions.values())
    assert q_sum == pytest.approx(1.0, abs=1e-3)

    # Check serialization
    d = summary.to_dict()
    assert d["spectral_norm"] == pytest.approx(10.0, rel=1e-3)


def test_inspect_checkpoint_spectral_profile_synthetic(tmp_path: Path):
    config = {
        "model_type": "sailor2",
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(16, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(32, 16, dtype=torch.bfloat16),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    profile = inspect_checkpoint_spectral_profile(tmp_path)

    assert profile.model_type == "sailor2"
    assert profile.num_hidden_layers == 1
    assert profile.hidden_size == 16
    assert profile.total_tensors == 4
    expected_params = (32 * 16) + (16 * 16) + 16 + (32 * 16)
    assert profile.total_parameters == expected_params

    assert "model.embed_tokens.weight" in profile.protected_tensor_names
    assert "lm_head.weight" in profile.protected_tensor_names
    assert "model.layers.0.self_attn.q_proj.weight" in profile.editable_tensor_names

    assert "self_attn.q_proj.weight" in profile.module_type_averages
    assert "input_layernorm.weight" in profile.module_type_averages
    q_stats = profile.module_type_averages["self_attn.q_proj.weight"]
    assert q_stats["count"] == 1
    assert "avg_condition_number" in q_stats
    assert "avg_effective_rank" in q_stats

    # Check to_dict JSON round-trip
    profile_dict = profile.to_dict()
    serialized = json.dumps(profile_dict, indent=2)
    deserialized = json.loads(serialized)
    assert deserialized["model_type"] == "sailor2"
    assert deserialized["total_tensors"] == 4
