from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from filbench_wsp.editing import (
    EditBudget,
    EditOp,
    EditPlan,
    EditValidationError,
    apply_edit_plan_to_checkpoint,
)
from filbench_wsp.editing.random_control import project_to_l2_and_linf_ball
from filbench_wsp.manifest import compute_checkpoint_hash


@pytest.fixture()
def synthetic_base_checkpoint(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "base_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(12345)
    tensors = {
        "model.embed_tokens.weight": torch.randn((64, 32), generator=gen).to(torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn((32, 32), generator=gen).to(
            torch.bfloat16
        ),
        "model.layers.0.self_attn.o_proj.weight": torch.randn((32, 32), generator=gen).to(
            torch.bfloat16
        ),
        "model.layers.0.mlp.down_proj.weight": torch.randn((32, 64), generator=gen).to(
            torch.bfloat16
        ),
        "model.layers.0.input_layernorm.weight": torch.ones((32,), dtype=torch.bfloat16),
        "lm_head.weight": torch.randn((64, 32), generator=gen).to(torch.bfloat16),
    }
    save_file(tensors, str(ckpt_dir / "model.safetensors"), metadata={"format": "pt"})
    (ckpt_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 32, "tie_word_embeddings": False}),
        encoding="utf-8",
    )
    # Nested subdirectory file to verify recursive copying
    nested_dir = ckpt_dir / "tokenizer"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>"}), encoding="utf-8"
    )
    return ckpt_dir


def test_rejects_protected_embeddings_and_tied_weights(
    synthetic_base_checkpoint: Path, tmp_path: Path
) -> None:
    for protected_name in ("model.embed_tokens.weight", "lm_head.weight"):
        with pytest.raises(EditValidationError, match="protected"):
            plan = EditPlan(
                plan_id="bad_protected",
                target_model="synthetic",
                hypothesis="should fail",
                ops=[
                    EditOp(
                        op_type="scale_tensor",
                        tensor_name=protected_name,
                        delta_scale=-0.05,
                    )
                ],
            )
            apply_edit_plan_to_checkpoint(
                plan=plan,
                base_checkpoint_dir=synthetic_base_checkpoint,
                output_checkpoint_dir=tmp_path / "out",
            )

    # Even if allow_embeddings_and_head=True, editing tied embeddings must be rejected
    tied_ckpt = tmp_path / "tied_ckpt"
    tied_ckpt.mkdir()
    save_file(
        {"model.embed_tokens.weight": torch.ones((16, 16), dtype=torch.bfloat16)},
        str(tied_ckpt / "model.safetensors"),
    )
    (tied_ckpt / "config.json").write_text(
        json.dumps({"tie_word_embeddings": True}), encoding="utf-8"
    )
    plan_tied = EditPlan(
        plan_id="bad_tied",
        target_model="synthetic",
        hypothesis="tied",
        budget=EditBudget(allow_embeddings_and_head=True),
        ops=[
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.embed_tokens.weight",
                delta_scale=0.05,
            )
        ],
    )
    with pytest.raises(EditValidationError, match="tie_word_embeddings=true"):
        apply_edit_plan_to_checkpoint(
            plan=plan_tied,
            base_checkpoint_dir=tied_ckpt,
            output_checkpoint_dir=tmp_path / "out_tied",
        )


def test_budget_from_dict_validates_keys_and_bounds() -> None:
    with pytest.raises(EditValidationError, match="Unknown keys in EditBudget"):
        EditPlan.from_dict(
            {
                "plan_id": "bad_budget_key",
                "target_model": "synthetic",
                "hypothesis": "unknown key",
                "budget": {"not_a_real_budget_field": 123},
                "ops": [
                    {
                        "op_type": "scale_tensor",
                        "tensor_name": "model.layers.0.self_attn.q_proj.weight",
                        "delta_scale": 0.05,
                    }
                ],
            }
        )

    with pytest.raises(EditValidationError, match="max_ops must be an integer >= 1"):
        EditPlan.from_dict(
            {
                "plan_id": "bad_budget_val",
                "target_model": "synthetic",
                "hypothesis": "zero max_ops",
                "budget": {"max_ops": 0},
                "ops": [
                    {
                        "op_type": "scale_tensor",
                        "tensor_name": "model.layers.0.self_attn.q_proj.weight",
                        "delta_scale": 0.05,
                    }
                ],
            }
        )


def test_rejects_unknown_tensor_and_out_of_bounds_slices(
    synthetic_base_checkpoint: Path, tmp_path: Path
) -> None:
    plan_unknown = EditPlan(
        plan_id="bad_unknown",
        target_model="synthetic",
        hypothesis="unknown tensor",
        ops=[
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.99.mlp.down_proj.weight",
                delta_scale=-0.05,
            )
        ],
    )
    with pytest.raises(EditValidationError, match="does not exist"):
        apply_edit_plan_to_checkpoint(
            plan=plan_unknown,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=tmp_path / "out_unknown",
        )

    plan_oob = EditPlan(
        plan_id="bad_oob",
        target_model="synthetic",
        hypothesis="oob slice",
        ops=[
            EditOp(
                op_type="scale_rows",
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                delta_scale=-0.05,
                row_start=16,
                row_end=40,
            )
        ],
    )
    with pytest.raises(EditValidationError, match="out of bounds"):
        apply_edit_plan_to_checkpoint(
            plan=plan_oob,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=tmp_path / "out_oob",
        )


def test_rejects_non_finite_and_excessive_magnitude(
    synthetic_base_checkpoint: Path, tmp_path: Path
) -> None:
    with pytest.raises(EditValidationError, match="Non-finite"):
        EditPlan.from_dict(
            {
                "plan_id": "nan_plan",
                "target_model": "synthetic",
                "hypothesis": "nan",
                "ops": [
                    {
                        "op_type": "scale_tensor",
                        "tensor_name": "model.layers.0.self_attn.q_proj.weight",
                        "delta_scale": float("nan"),
                    }
                ],
            }
        )

    plan_excess_frob = EditPlan(
        plan_id="excess_frob",
        target_model="synthetic",
        hypothesis="stacking exceeds frob budget",
        ops=[
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                delta_scale=0.08,
            ),
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                delta_scale=0.08,
            ),
        ],
    )
    with pytest.raises(EditValidationError, match="relative Frobenius norm"):
        apply_edit_plan_to_checkpoint(
            plan=plan_excess_frob,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=tmp_path / "out_frob",
        )


def test_l2_and_linf_alternating_projection_converges() -> None:
    # Construct a spiky vector where simple rescaling to target_frob would exceed max_abs_entry_limit
    vec = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    target_frob = 3.0
    max_abs_limit = 1.5  # single-step clamp+rescale would push the spike above 1.5
    projected = project_to_l2_and_linf_ball(
        delta=vec,
        target_frobenius_fp32=target_frob,
        max_abs_entry_limit=max_abs_limit,
        tensor_name="spiky_test",
    )
    assert float(torch.linalg.vector_norm(projected).item()) == pytest.approx(
        target_frob, rel=1e-5
    )
    assert float(torch.max(torch.abs(projected)).item()) <= max_abs_limit


def test_rejects_bf16_underflow_noop(synthetic_base_checkpoint: Path, tmp_path: Path) -> None:
    plan_tiny = EditPlan(
        plan_id="tiny_noop",
        target_model="synthetic",
        hypothesis="tiny scale rounds to zero in bf16",
        ops=[
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.0.input_layernorm.weight",
                delta_scale=1e-6,
            )
        ],
    )
    with pytest.raises(EditValidationError, match="rounded to zero"):
        apply_edit_plan_to_checkpoint(
            plan=plan_tiny,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=tmp_path / "out_tiny",
        )


def test_refuses_to_overwrite_base_or_existing_output(
    synthetic_base_checkpoint: Path, tmp_path: Path
) -> None:
    plan = EditPlan(
        plan_id="valid_plan",
        target_model="synthetic",
        hypothesis="valid",
        ops=[
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                delta_scale=-0.05,
            )
        ],
    )
    with pytest.raises(EditValidationError, match="Refusing to write"):
        apply_edit_plan_to_checkpoint(
            plan=plan,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=synthetic_base_checkpoint,
        )

    out_dir = tmp_path / "existing_out"
    apply_edit_plan_to_checkpoint(
        plan=plan,
        base_checkpoint_dir=synthetic_base_checkpoint,
        output_checkpoint_dir=out_dir,
    )
    with pytest.raises(FileExistsError):
        apply_edit_plan_to_checkpoint(
            plan=plan,
            base_checkpoint_dir=synthetic_base_checkpoint,
            output_checkpoint_dir=out_dir,
            force=False,
        )


def test_applies_all_four_operators_copies_subdirs_and_matches_random_control(
    synthetic_base_checkpoint: Path, tmp_path: Path
) -> None:
    base_hash_initial = compute_checkpoint_hash(synthetic_base_checkpoint)

    plan = EditPlan(
        plan_id="four_op_test",
        target_model="synthetic",
        hypothesis="test all 4 structured operators",
        ops=[
            EditOp(
                op_type="scale_rows",
                tensor_name="model.layers.0.self_attn.q_proj.weight",
                delta_scale=-0.08,
                row_start=0,
                row_end=8,
            ),
            EditOp(
                op_type="scale_cols",
                tensor_name="model.layers.0.self_attn.o_proj.weight",
                delta_scale=0.06,
                col_start=8,
                col_end=16,
            ),
            EditOp(
                op_type="spectral_band_scale",
                tensor_name="model.layers.0.mlp.down_proj.weight",
                delta_scale=-0.10,
                rank_start=0,
                rank_end=4,
            ),
            EditOp(
                op_type="scale_tensor",
                tensor_name="model.layers.0.input_layernorm.weight",
                delta_scale=0.04,
            ),
        ],
        budget=EditBudget(),
    )

    edited_dir = tmp_path / "ckpt_edited"
    random_dir = tmp_path / "ckpt_random"

    rep_edit = apply_edit_plan_to_checkpoint(
        plan=plan,
        base_checkpoint_dir=synthetic_base_checkpoint,
        output_checkpoint_dir=edited_dir,
        mode="llm_edited",
    )
    rep_rand = apply_edit_plan_to_checkpoint(
        plan=plan,
        base_checkpoint_dir=synthetic_base_checkpoint,
        output_checkpoint_dir=random_dir,
        mode="random_control",
        random_seed=42,
    )

    # 1. Base checkpoint must be completely untouched
    assert compute_checkpoint_hash(synthetic_base_checkpoint) == base_hash_initial
    assert rep_edit.base_checkpoint_sha256 == base_hash_initial
    assert rep_rand.base_checkpoint_sha256 == base_hash_initial

    # 2. Nested subdirectory files must be copied into both output checkpoints
    assert (edited_dir / "tokenizer" / "special_tokens_map.json").exists()
    assert (random_dir / "tokenizer" / "special_tokens_map.json").exists()

    # 3. Edited and random checkpoints must differ from base and from each other
    assert rep_edit.output_checkpoint_sha256 != base_hash_initial
    assert rep_rand.output_checkpoint_sha256 != base_hash_initial
    assert rep_edit.output_checkpoint_sha256 != rep_rand.output_checkpoint_sha256

    # 4. Exactly the same 4 tensors edited, and fp32 Frobenius norms match closely
    assert rep_edit.num_edited_tensors == 4
    assert rep_rand.num_edited_tensors == 4
    for m_edit, m_rand in zip(rep_edit.tensor_metrics, rep_rand.tensor_metrics):
        assert m_edit.tensor_name == m_rand.tensor_name
        assert m_edit.fp32_delta_frobenius == pytest.approx(
            m_rand.fp32_delta_frobenius, rel=1e-4
        )
        assert m_rand.fp32_rel_linf <= plan.budget.max_rel_linf

    # 5. Untouched tensors in output checkpoints must be bitwise identical to base
    with (
        safe_open(synthetic_base_checkpoint / "model.safetensors", framework="pt") as f_base,
        safe_open(edited_dir / "model.safetensors", framework="pt") as f_edit,
        safe_open(random_dir / "model.safetensors", framework="pt") as f_rand,
    ):
        for untouched in ("model.embed_tokens.weight", "lm_head.weight"):
            assert torch.equal(f_base.get_tensor(untouched), f_edit.get_tensor(untouched))
            assert torch.equal(f_base.get_tensor(untouched), f_rand.get_tensor(untouched))
