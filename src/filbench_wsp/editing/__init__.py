"""Fail-closed weight-editing safety harness and random-control generation."""

from __future__ import annotations

from filbench_wsp.editing.applier import apply_edit_plan_to_checkpoint, compute_op_delta_fp32
from filbench_wsp.editing.random_control import (
    build_matched_random_tensor_delta_fp32,
    project_to_l2_and_linf_ball,
)
from filbench_wsp.editing.schema import (
    DEFAULT_PROTECTED_TENSORS,
    VALID_OP_TYPES,
    EditBudget,
    EditExecutionReport,
    EditOp,
    EditPlan,
    EditValidationError,
    OpType,
    TensorEditMetrics,
)

__all__ = [
    "DEFAULT_PROTECTED_TENSORS",
    "VALID_OP_TYPES",
    "EditBudget",
    "EditExecutionReport",
    "EditOp",
    "EditPlan",
    "EditValidationError",
    "OpType",
    "TensorEditMetrics",
    "apply_edit_plan_to_checkpoint",
    "build_matched_random_tensor_delta_fp32",
    "compute_op_delta_fp32",
    "project_to_l2_and_linf_ball",
]
