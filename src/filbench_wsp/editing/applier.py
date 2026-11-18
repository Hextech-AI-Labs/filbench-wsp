from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from filbench_wsp.editing.random_control import build_matched_random_tensor_delta_fp32
from filbench_wsp.editing.schema import (
    EditBudget,
    EditExecutionReport,
    EditOp,
    EditPlan,
    EditValidationError,
    TensorEditMetrics,
)
from filbench_wsp.manifest import compute_checkpoint_hash

EPS = 1e-12
# bfloat16 has 7 mantissa bits (~2^-7 = 0.78125% relative quantization step).
# Rounding (base_fp32 + delta_fp32) back to bfloat16 can shift the realized perturbation
# norm by up to ~1% relative to the pre-rounding fp32 budget cap.
BF16_BUDGET_SLACK = 1e-2


def compute_op_delta_fp32(base_weight_fp32: torch.Tensor, op: EditOp) -> torch.Tensor:
    """Compute the exact fp32 weight perturbation Delta W for a single validated EditOp."""
    op.validate_against_shape(tuple(base_weight_fp32.shape))
    delta = torch.zeros_like(base_weight_fp32, dtype=torch.float32)
    scale = float(op.delta_scale)

    if op.op_type == "scale_tensor":
        delta = scale * base_weight_fp32

    elif op.op_type == "scale_rows":
        if op.row_start is None or op.row_end is None:
            raise EditValidationError("scale_rows requires row_start and row_end")
        if base_weight_fp32.ndim == 1:
            delta[op.row_start : op.row_end] = (
                scale * base_weight_fp32[op.row_start : op.row_end]
            )
        else:
            delta[op.row_start : op.row_end, :] = (
                scale * base_weight_fp32[op.row_start : op.row_end, :]
            )

    elif op.op_type == "scale_cols":
        if op.col_start is None or op.col_end is None:
            raise EditValidationError("scale_cols requires col_start and col_end")
        delta[:, op.col_start : op.col_end] = (
            scale * base_weight_fp32[:, op.col_start : op.col_end]
        )

    elif op.op_type == "spectral_band_scale":
        if op.rank_start is None or op.rank_end is None:
            raise EditValidationError("spectral_band_scale requires rank_start and rank_end")
        u, s, vh = torch.linalg.svd(base_weight_fp32, full_matrices=False)
        u_band = u[:, op.rank_start : op.rank_end]
        s_band = s[op.rank_start : op.rank_end]
        vh_band = vh[op.rank_start : op.rank_end, :]
        delta = scale * ((u_band * s_band.unsqueeze(0)) @ vh_band)

    else:
        raise EditValidationError(f"Unsupported op_type: {op.op_type!r}")

    if not bool(torch.isfinite(delta).all().item()):
        raise EditValidationError(
            f"Operation {op.op_type!r} on {op.tensor_name!r} produced non-finite values"
        )
    return delta


def compute_and_validate_tensor_edit(
    tensor_name: str,
    base_weight: torch.Tensor,
    ops: list[EditOp],
    budget: EditBudget,
    mode: Literal["llm_edited", "random_control"] = "llm_edited",
    random_seed: int | None = None,
) -> tuple[torch.Tensor, TensorEditMetrics]:
    """Compute new bfloat16 tensor and validate all fp32 and bf16 safety invariants."""
    if not bool(torch.isfinite(base_weight).all().item()):
        raise EditValidationError(f"Base tensor {tensor_name!r} contains non-finite values")

    base_fp32 = base_weight.to(torch.float32)
    base_frob = float(torch.linalg.vector_norm(base_fp32).item())
    base_linf = float(torch.max(torch.abs(base_fp32)).item())

    llm_delta_fp32 = torch.zeros_like(base_fp32, dtype=torch.float32)
    for op in ops:
        llm_delta_fp32 = llm_delta_fp32 + compute_op_delta_fp32(base_fp32, op)

    if mode == "llm_edited":
        delta_fp32 = llm_delta_fp32
    elif mode == "random_control":
        if random_seed is None:
            raise EditValidationError("random_seed is required when mode='random_control'")
        target_frob = float(torch.linalg.vector_norm(llm_delta_fp32).item())
        max_abs_limit = budget.max_rel_linf * (base_linf + EPS)
        delta_fp32 = build_matched_random_tensor_delta_fp32(
            base_weight_fp32=base_fp32,
            ops=ops,
            target_frobenius_fp32=target_frob,
            max_abs_entry_limit=max_abs_limit,
            master_seed=random_seed,
        )
    else:
        raise EditValidationError(f"Invalid mode: {mode!r}")

    if not bool(torch.isfinite(delta_fp32).all().item()):
        raise EditValidationError(f"Combined delta for {tensor_name!r} contains non-finite values")

    fp32_delta_frob = float(torch.linalg.vector_norm(delta_fp32).item())
    fp32_rel_frob = fp32_delta_frob / (base_frob + EPS)
    fp32_delta_linf = float(torch.max(torch.abs(delta_fp32)).item())
    fp32_rel_linf = fp32_delta_linf / (base_linf + EPS)

    if fp32_rel_frob > budget.max_rel_frobenius + 1e-6:
        raise EditValidationError(
            f"Tensor {tensor_name!r} fp32 relative Frobenius norm {fp32_rel_frob:.6f} "
            f"exceeds budget.max_rel_frobenius={budget.max_rel_frobenius:.6f}"
        )
    if fp32_rel_linf > budget.max_rel_linf + 1e-6:
        raise EditValidationError(
            f"Tensor {tensor_name!r} fp32 relative L_inf norm {fp32_rel_linf:.6f} "
            f"exceeds budget.max_rel_linf={budget.max_rel_linf:.6f}"
        )

    edited_fp32 = base_fp32 + delta_fp32
    if not bool(torch.isfinite(edited_fp32).all().item()):
        raise EditValidationError(f"Edited fp32 tensor {tensor_name!r} contains non-finite values")

    edited_out = edited_fp32.to(base_weight.dtype)
    if not bool(torch.isfinite(edited_out).all().item()):
        raise EditValidationError(
            f"Edited tensor {tensor_name!r} overflowed after casting to {base_weight.dtype}"
        )

    actual_delta_fp32 = edited_out.to(torch.float32) - base_fp32
    bf16_l0 = int(torch.count_nonzero(edited_out != base_weight).item())
    numel = int(base_weight.numel())
    bf16_l0_frac = bf16_l0 / max(1, numel)

    if budget.require_bf16_nonzero_change and bf16_l0 == 0:
        raise EditValidationError(
            f"Edit on {tensor_name!r} rounded to zero in {base_weight.dtype} (no-op); "
            f"increase |delta_scale| above bfloat16 quantization floor"
        )

    bf16_delta_frob = float(torch.linalg.vector_norm(actual_delta_fp32).item())
    bf16_rel_frob = bf16_delta_frob / (base_frob + EPS)
    bf16_delta_linf = float(torch.max(torch.abs(actual_delta_fp32)).item())
    bf16_rel_linf = bf16_delta_linf / (base_linf + EPS)

    if bf16_rel_frob > budget.max_rel_frobenius + BF16_BUDGET_SLACK:
        raise EditValidationError(
            f"Tensor {tensor_name!r} post-rounding relative Frobenius norm {bf16_rel_frob:.6f} "
            f"exceeds budget={budget.max_rel_frobenius:.6f}"
        )
    if bf16_rel_linf > budget.max_rel_linf + BF16_BUDGET_SLACK:
        raise EditValidationError(
            f"Tensor {tensor_name!r} post-rounding relative L_inf norm {bf16_rel_linf:.6f} "
            f"exceeds budget={budget.max_rel_linf:.6f}"
        )

    metrics = TensorEditMetrics(
        tensor_name=tensor_name,
        shape=list(base_weight.shape),
        numel=numel,
        num_ops_applied=len(ops),
        base_frobenius=base_frob,
        base_linf=base_linf,
        fp32_delta_frobenius=fp32_delta_frob,
        fp32_rel_frobenius=fp32_rel_frob,
        fp32_delta_linf=fp32_delta_linf,
        fp32_rel_linf=fp32_rel_linf,
        bf16_l0_changed_elements=bf16_l0,
        bf16_l0_fraction=bf16_l0_frac,
        bf16_delta_frobenius=bf16_delta_frob,
        bf16_rel_frobenius=bf16_rel_frob,
        bf16_delta_linf=bf16_delta_linf,
        bf16_rel_linf=bf16_rel_linf,
    )
    return edited_out, metrics


def _check_tied_embeddings_policy(base_dir: Path, targeted_tensors: set[str]) -> None:
    config_path = base_dir / "config.json"
    if not config_path.exists():
        return
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if bool(cfg.get("tie_word_embeddings", False)):
        tied_pair = {"model.embed_tokens.weight", "lm_head.weight"}
        touched_tied = targeted_tensors & tied_pair
        if touched_tied:
            raise EditValidationError(
                f"Checkpoint {base_dir} has tie_word_embeddings=true; editing {sorted(touched_tied)} "
                f"is forbidden because it would desynchronize tied input/output embeddings."
            )


def _verify_saved_checkpoint(
    out_dir: Path,
    st_files: list[Path],
    expected_keys_by_file: dict[str, list[str]],
    edited_tensors: dict[str, torch.Tensor],
) -> None:
    """Verify every output .safetensors shard is loadable and edited tensors match bitwise."""
    for st_path in st_files:
        out_st_path = out_dir / st_path.name
        if not out_st_path.exists():
            raise RuntimeError(f"Expected output safetensors file missing: {out_st_path}")
        with safe_open(out_st_path, framework="pt", device="cpu") as f:
            actual_keys = list(f.keys())
            expected_keys = expected_keys_by_file[st_path.name]
            if set(actual_keys) != set(expected_keys):
                raise RuntimeError(
                    f"Key mismatch in saved {out_st_path}: expected {len(expected_keys)} keys, got {len(actual_keys)}"
                )
            # Always load at least one tensor per shard plus all edited tensors to verify integrity
            probed = False
            for key in actual_keys:
                if key in edited_tensors:
                    loaded = f.get_tensor(key)
                    if not torch.equal(loaded, edited_tensors[key]):
                        raise RuntimeError(
                            f"Saved tensor {key!r} in {out_st_path} does not match validated tensor in memory"
                        )
                    probed = True
            if not probed and actual_keys:
                _ = f.get_tensor(actual_keys[0])


def apply_edit_plan_to_checkpoint(
    plan: EditPlan,
    base_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    mode: Literal["llm_edited", "random_control"] = "llm_edited",
    random_seed: int | None = None,
    force: bool = False,
) -> EditExecutionReport:
    """Validate and materialize an edited or random-control checkpoint non-destructively."""
    plan.validate_syntax()

    base_dir = base_checkpoint_dir.resolve()
    out_dir = output_checkpoint_dir.resolve()

    if out_dir == base_dir or base_dir in out_dir.parents or out_dir in base_dir.parents:
        raise EditValidationError(
            f"Refusing to write output checkpoint to {out_dir}: overlaps with base_checkpoint_dir {base_dir}"
        )

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output checkpoint directory {out_dir} already exists and is non-empty. Pass force=True to overwrite."
        )

    st_files = sorted(base_dir.glob("*.safetensors"))
    if not st_files:
        raise EditValidationError(f"No .safetensors files found in {base_dir}")

    ops_by_tensor: dict[str, list[EditOp]] = defaultdict(list)
    for op in plan.ops:
        ops_by_tensor[op.tensor_name].append(op)

    _check_tied_embeddings_policy(base_dir, set(ops_by_tensor.keys()))

    # Single header + targeted-tensor pass over base .safetensors files
    inventory_shapes: dict[str, tuple[int, ...]] = {}
    expected_keys_by_file: dict[str, list[str]] = {}
    metadata_by_file: dict[str, dict[str, str]] = {}
    targeted_base_tensors: dict[str, torch.Tensor] = {}

    for st_path in st_files:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            expected_keys_by_file[st_path.name] = keys
            metadata_by_file[st_path.name] = f.metadata() or {"format": "pt"}
            for key in keys:
                s = f.get_slice(key)
                inventory_shapes[key] = tuple(s.get_shape())
                if key in ops_by_tensor:
                    targeted_base_tensors[key] = f.get_tensor(key)

    for tensor_name, ops in ops_by_tensor.items():
        if tensor_name not in inventory_shapes:
            raise EditValidationError(
                f"Target tensor {tensor_name!r} does not exist in checkpoint {base_dir}"
            )
        shape = inventory_shapes[tensor_name]
        for op in ops:
            op.validate_against_shape(shape)

    base_sha_before = compute_checkpoint_hash(base_dir)

    edited_tensors: dict[str, torch.Tensor] = {}
    metrics_list: list[TensorEditMetrics] = []

    for tensor_name in sorted(ops_by_tensor.keys()):
        base_tensor = targeted_base_tensors[tensor_name]
        new_tensor, metrics = compute_and_validate_tensor_edit(
            tensor_name=tensor_name,
            base_weight=base_tensor,
            ops=ops_by_tensor[tensor_name],
            budget=plan.budget,
            mode=mode,
            random_seed=random_seed,
        )
        edited_tensors[tensor_name] = new_tensor
        metrics_list.append(metrics)

    # All validations succeeded; materialize output_checkpoint_dir cleanly
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recursively copy all non-.safetensors files (configs, tokenizers, nested subfolders)
    for item in base_dir.rglob("*"):
        if item.is_file() and item.suffix != ".safetensors":
            rel_path = item.relative_to(base_dir)
            dest_path = out_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_path)

    for st_path in st_files:
        tensors_to_save: dict[str, torch.Tensor] = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in expected_keys_by_file[st_path.name]:
                if key in edited_tensors:
                    tensors_to_save[key] = edited_tensors[key]
                else:
                    tensors_to_save[key] = f.get_tensor(key)
        save_file(
            tensors_to_save,
            str(out_dir / st_path.name),
            metadata=metadata_by_file[st_path.name],
        )

    _verify_saved_checkpoint(
        out_dir=out_dir,
        st_files=st_files,
        expected_keys_by_file=expected_keys_by_file,
        edited_tensors=edited_tensors,
    )

    base_sha_after = compute_checkpoint_hash(base_dir)
    if base_sha_after != base_sha_before:
        raise RuntimeError(
            f"CRITICAL: Base checkpoint hash changed during edit application! "
            f"Before={base_sha_before}, After={base_sha_after}"
        )

    output_sha = compute_checkpoint_hash(out_dir)

    global_fp32_sq = sum(m.fp32_delta_frobenius**2 for m in metrics_list)
    global_bf16_sq = sum(m.bf16_delta_frobenius**2 for m in metrics_list)
    total_changed_bf16 = sum(m.bf16_l0_changed_elements for m in metrics_list)

    return EditExecutionReport(
        plan_id=plan.plan_id,
        mode=mode,
        seed=random_seed,
        base_checkpoint_dir=str(base_dir),
        output_checkpoint_dir=str(out_dir),
        base_checkpoint_sha256=base_sha_after,
        output_checkpoint_sha256=output_sha,
        total_tensors_in_model=len(inventory_shapes),
        num_edited_tensors=len(metrics_list),
        total_changed_parameters_bf16=total_changed_bf16,
        global_fp32_delta_frobenius=global_fp32_sq**0.5,
        global_bf16_delta_frobenius=global_bf16_sq**0.5,
        global_max_rel_frobenius_fp32=max(m.fp32_rel_frobenius for m in metrics_list),
        global_max_rel_linf_fp32=max(m.fp32_rel_linf for m in metrics_list),
        tensor_metrics=metrics_list,
    )
