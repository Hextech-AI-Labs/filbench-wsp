from __future__ import annotations

import hashlib
import torch

from filbench_wsp.editing.schema import EditOp, EditValidationError

EPS = 1e-12
MAX_PROJECTION_ITERS = 64


def _derive_tensor_seed(master_seed: int, tensor_name: str, op_index: int) -> int:
    payload = f"{master_seed}:{tensor_name}:{op_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & 0x7FFF_FFFF_FFFF_FFFF


def _rademacher(shape: tuple[int, ...], gen: torch.Generator, device: torch.device) -> torch.Tensor:
    bits = torch.randint(0, 2, shape, generator=gen, device=device, dtype=torch.int64)
    return bits.to(torch.float32).mul_(2.0).sub_(1.0)


def _random_orthonormal_columns(
    num_rows: int,
    num_cols: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    gaussian = torch.randn((num_rows, num_cols), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    diag = torch.diagonal(r, 0)
    signs = torch.where(diag >= 0.0, torch.ones_like(diag), -torch.ones_like(diag))
    return q * signs.unsqueeze(0)


def build_random_op_direction_fp32(
    base_weight_fp32: torch.Tensor,
    op: EditOp,
    master_seed: int,
    op_index: int,
) -> torch.Tensor:
    """Construct an unnormalized structure-matched random perturbation for a single EditOp."""
    op.validate_against_shape(tuple(base_weight_fp32.shape))
    device = base_weight_fp32.device
    gen = torch.Generator(device=device)
    gen.manual_seed(_derive_tensor_seed(master_seed, op.tensor_name, op_index))

    shape = tuple(base_weight_fp32.shape)
    delta_rand = torch.zeros_like(base_weight_fp32, dtype=torch.float32)

    if op.op_type == "scale_tensor":
        if len(shape) == 1:
            signs = _rademacher(shape, gen, device)
            delta_rand = float(op.delta_scale) * (base_weight_fp32 * signs)
        else:
            row_signs = _rademacher((shape[0], 1), gen, device)
            col_signs = _rademacher((1, shape[1]), gen, device)
            delta_rand = float(op.delta_scale) * (base_weight_fp32 * row_signs * col_signs)

    elif op.op_type == "scale_rows":
        if op.row_start is None or op.row_end is None:
            raise EditValidationError("scale_rows requires row_start and row_end")
        num_rows = op.row_end - op.row_start
        perm = torch.randperm(shape[0], generator=gen, device=device)[:num_rows]
        if len(shape) == 1:
            signs = _rademacher((num_rows,), gen, device)
            delta_rand[perm] = float(op.delta_scale) * (base_weight_fp32[perm] * signs)
        else:
            signs = _rademacher((num_rows, 1), gen, device)
            delta_rand[perm, :] = float(op.delta_scale) * (base_weight_fp32[perm, :] * signs)

    elif op.op_type == "scale_cols":
        if op.col_start is None or op.col_end is None:
            raise EditValidationError("scale_cols requires col_start and col_end")
        num_cols = op.col_end - op.col_start
        perm = torch.randperm(shape[1], generator=gen, device=device)[:num_cols]
        signs = _rademacher((1, num_cols), gen, device)
        delta_rand[:, perm] = float(op.delta_scale) * (base_weight_fp32[:, perm] * signs)

    elif op.op_type == "spectral_band_scale":
        if op.rank_start is None or op.rank_end is None:
            raise EditValidationError("spectral_band_scale requires rank_start and rank_end")
        band_rank = op.rank_end - op.rank_start
        s = torch.linalg.svdvals(base_weight_fp32)
        band_sigma = s[op.rank_start : op.rank_end]
        u_rand = _random_orthonormal_columns(shape[0], band_rank, gen, device)
        v_rand = _random_orthonormal_columns(shape[1], band_rank, gen, device)
        delta_rand = float(op.delta_scale) * ((u_rand * band_sigma.unsqueeze(0)) @ v_rand.T)

    else:
        raise EditValidationError(f"Unsupported op_type {op.op_type!r} in random control")

    return delta_rand


def project_to_l2_and_linf_ball(
    delta: torch.Tensor,
    target_frobenius_fp32: float,
    max_abs_entry_limit: float,
    tensor_name: str,
) -> torch.Tensor:
    """Project a perturbation onto {||x||_F == target_frobenius_fp32} intersect {||x||_inf <= max_abs_entry_limit}.

    Uses alternating projections (clamp to L_inf box, rescale to L_2 sphere) and fails closed
    if the target Frobenius norm is mathematically infeasible on the active support.
    """
    if target_frobenius_fp32 <= EPS:
        raise EditValidationError(
            f"Target Frobenius norm for {tensor_name!r} must be positive, got {target_frobenius_fp32}"
        )
    if max_abs_entry_limit <= EPS:
        raise EditValidationError(
            f"max_abs_entry_limit for {tensor_name!r} must be positive, got {max_abs_entry_limit}"
        )

    active_support = int(torch.count_nonzero(delta != 0.0).item())
    max_feasible_frob = (active_support**0.5) * max_abs_entry_limit
    if target_frobenius_fp32 > max_feasible_frob * (1.0 - 1e-5):
        raise EditValidationError(
            f"Random control for {tensor_name!r} cannot satisfy both L2={target_frobenius_fp32:.6f} "
            f"and L_inf<={max_abs_entry_limit:.6f} on support size {active_support} "
            f"(max feasible L2 is {max_feasible_frob:.6f})"
        )

    init_frob = float(torch.linalg.vector_norm(delta).item())
    if init_frob <= EPS:
        raise EditValidationError(
            f"Random control perturbation for {tensor_name!r} has near-zero initial Frobenius norm ({init_frob})"
        )

    x = delta * (target_frobenius_fp32 / init_frob)
    clamp_cap = max_abs_entry_limit * 0.995

    for _ in range(MAX_PROJECTION_ITERS):
        cur_linf = float(torch.max(torch.abs(x)).item())
        cur_frob = float(torch.linalg.vector_norm(x).item())
        if (
            cur_linf <= max_abs_entry_limit
            and abs(cur_frob - target_frobenius_fp32) <= 1e-5 * target_frobenius_fp32
        ):
            return x

        x = torch.clamp(x, min=-clamp_cap, max=clamp_cap)
        clamped_frob = float(torch.linalg.vector_norm(x).item())
        if clamped_frob <= EPS:
            raise EditValidationError(
                f"Random control projection collapsed to zero norm for {tensor_name!r}"
            )
        x = x * (target_frobenius_fp32 / clamped_frob)

    final_linf = float(torch.max(torch.abs(x)).item())
    final_frob = float(torch.linalg.vector_norm(x).item())
    if (
        final_linf > max_abs_entry_limit
        or abs(final_frob - target_frobenius_fp32) > 1e-4 * target_frobenius_fp32
    ):
        raise EditValidationError(
            f"Random control alternating L2/L_inf projection failed to converge for {tensor_name!r}: "
            f"L2={final_frob:.6f} (target={target_frobenius_fp32:.6f}), "
            f"L_inf={final_linf:.6f} (limit={max_abs_entry_limit:.6f})"
        )
    return x


def build_matched_random_tensor_delta_fp32(
    base_weight_fp32: torch.Tensor,
    ops: list[EditOp],
    target_frobenius_fp32: float,
    max_abs_entry_limit: float,
    master_seed: int,
) -> torch.Tensor:
    """Generate a structure- and Frobenius-norm-matched random perturbation for a tensor."""
    if not ops:
        raise EditValidationError("ops must be non-empty in build_matched_random_tensor_delta_fp32")

    raw_delta = torch.zeros_like(base_weight_fp32, dtype=torch.float32)
    for idx, op in enumerate(ops):
        raw_delta = raw_delta + build_random_op_direction_fp32(
            base_weight_fp32=base_weight_fp32,
            op=op,
            master_seed=master_seed,
            op_index=idx,
        )

    return project_to_l2_and_linf_ball(
        delta=raw_delta,
        target_frobenius_fp32=target_frobenius_fp32,
        max_abs_entry_limit=max_abs_entry_limit,
        tensor_name=ops[0].tensor_name,
    )
