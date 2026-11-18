"""Safety tests for the tie-safe scale_tied_rows operator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from filbench_wsp.editing import EditPlan, EditValidationError, apply_edit_plan_to_checkpoint
from filbench_wsp.manifest import compute_checkpoint_hash


def make_ckpt(tmp_path: Path, tie: bool = True, with_lm: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    embed = torch.randn(64, 16, generator=g).to(torch.bfloat16)
    tensors = {
        "model.embed_tokens.weight": embed.clone(),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 32, generator=g).to(torch.bfloat16),
        "model.norm.weight": torch.ones(16, dtype=torch.bfloat16),
    }
    if with_lm:
        tensors["lm_head.weight"] = embed.clone()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"tie_word_embeddings": tie, "num_hidden_layers": 1}), encoding="utf-8"
    )
    return tmp_path


def tied_plan(budget_extra: dict | None = None, **op_extra) -> EditPlan:
    op = {
        "op_type": "scale_tied_rows",
        "tensor_name": "model.embed_tokens.weight",
        "delta_scale": 0.05,
        "row_indices": [1, 5, 9],
        "rationale": "test",
    }
    op.update(op_extra)
    d = {
        "plan_id": "tied_test",
        "target_model": "test",
        "hypothesis": "test",
        "ops": [op],
        "budget": {"allow_tied_embedding_rows": True, **(budget_extra or {})},
    }
    return EditPlan.from_dict(d)


def read_tensor(ckpt: Path, key: str) -> torch.Tensor:
    with safe_open(ckpt / "model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def test_tied_edit_preserves_tie_and_base(tmp_path):
    base = make_ckpt(tmp_path / "base")
    out = tmp_path / "out"
    sha_before = compute_checkpoint_hash(base)
    rep = apply_edit_plan_to_checkpoint(tied_plan(), base, out)
    e, l = read_tensor(out, "model.embed_tokens.weight"), read_tensor(out, "lm_head.weight")
    assert torch.equal(e, l)
    assert not torch.equal(e, read_tensor(base, "model.embed_tokens.weight"))
    assert rep.num_edited_tensors == 2
    assert compute_checkpoint_hash(base) == sha_before


def test_random_control_preserves_tie(tmp_path):
    base = make_ckpt(tmp_path / "base")
    out = tmp_path / "out"
    apply_edit_plan_to_checkpoint(tied_plan(), base, out, mode="random_control", random_seed=7)
    assert torch.equal(read_tensor(out, "model.embed_tokens.weight"), read_tensor(out, "lm_head.weight"))


def test_flag_off_rejected():
    d = tied_plan().to_dict()
    d["budget"] = {}
    with pytest.raises(EditValidationError):
        EditPlan.from_dict(d)


def test_untied_checkpoint_rejected(tmp_path):
    base = make_ckpt(tmp_path / "base", tie=False)
    with pytest.raises(EditValidationError):
        apply_edit_plan_to_checkpoint(tied_plan(), base, tmp_path / "out")


def test_missing_lm_head_ok(tmp_path):
    base = make_ckpt(tmp_path / "base", with_lm=False)
    out = tmp_path / "out"
    rep = apply_edit_plan_to_checkpoint(tied_plan(), base, out)
    assert rep.num_edited_tensors == 1
    assert not torch.equal(read_tensor(out, "model.embed_tokens.weight"),
                           read_tensor(base, "model.embed_tokens.weight"))


def test_explicit_lm_head_rejected():
    with pytest.raises(EditValidationError):
        tied_plan(tensor_name="lm_head.weight")


def test_mixing_ops_on_embed_rejected(tmp_path):
    base = make_ckpt(tmp_path / "base")
    d = tied_plan().to_dict()
    d["ops"].append({"op_type": "scale_tensor", "tensor_name": "model.embed_tokens.weight",
                     "delta_scale": 0.02})
    d["budget"]["allow_embeddings_and_head"] = True
    with pytest.raises(EditValidationError):
        apply_edit_plan_to_checkpoint(EditPlan.from_dict(d), base, tmp_path / "out")


def test_row_indices_validation():
    with pytest.raises(EditValidationError):
        tied_plan(row_indices=[])
    with pytest.raises(EditValidationError):
        tied_plan(row_indices=[1, 1])
    with pytest.raises(EditValidationError):
        tied_plan(row_indices=[-1])
    with pytest.raises(EditValidationError):
        tied_plan(row_indices=list(range(600)))


def test_row_indices_out_of_bounds(tmp_path):
    base = make_ckpt(tmp_path / "base")
    with pytest.raises(EditValidationError):
        apply_edit_plan_to_checkpoint(tied_plan(row_indices=[1, 999]), base, tmp_path / "out")


def test_ordinary_ops_still_blocked_on_tied_embeddings(tmp_path):
    base = make_ckpt(tmp_path / "base")
    d = {"plan_id": "x", "target_model": "t", "hypothesis": "h",
         "ops": [{"op_type": "scale_tensor", "tensor_name": "model.embed_tokens.weight",
                  "delta_scale": 0.02}],
         "budget": {"allow_embeddings_and_head": True}}
    with pytest.raises(EditValidationError):
        apply_edit_plan_to_checkpoint(EditPlan.from_dict(d), base, tmp_path / "out")