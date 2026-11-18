from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OpType = Literal[
    "scale_tensor",
    "scale_rows",
    "scale_cols",
    "spectral_band_scale",
]

VALID_OP_TYPES: frozenset[str] = frozenset(
    {
        "scale_tensor",
        "scale_rows",
        "scale_cols",
        "spectral_band_scale",
    }
)

DEFAULT_PROTECTED_TENSORS: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.embed_positions.weight",
    "transformer.wte.weight",
    "transformer.wpe.weight",
)


class EditValidationError(ValueError):
    """Raised when an untrusted edit operation or plan violates safety invariants."""


@dataclass(frozen=True)
class EditBudget:
    """Fail-closed safety limits for untrusted weight-edit plans."""

    max_ops: int = 32
    max_edited_tensors: int = 24
    max_abs_delta_scale: float = 0.15
    max_rel_frobenius: float = 0.10
    max_rel_linf: float = 0.25
    allow_embeddings_and_head: bool = False
    require_bf16_nonzero_change: bool = True
    protected_tensors: tuple[str, ...] = DEFAULT_PROTECTED_TENSORS

    def validate(self) -> None:
        if not isinstance(self.max_ops, int) or isinstance(self.max_ops, bool) or self.max_ops < 1:
            raise EditValidationError("max_ops must be an integer >= 1")
        if (
            not isinstance(self.max_edited_tensors, int)
            or isinstance(self.max_edited_tensors, bool)
            or self.max_edited_tensors < 1
        ):
            raise EditValidationError("max_edited_tensors must be an integer >= 1")
        if not math.isfinite(self.max_abs_delta_scale) or not (0.0 < self.max_abs_delta_scale <= 1.0):
            raise EditValidationError("max_abs_delta_scale must be in (0.0, 1.0]")
        if not math.isfinite(self.max_rel_frobenius) or not (0.0 < self.max_rel_frobenius <= 1.0):
            raise EditValidationError("max_rel_frobenius must be in (0.0, 1.0]")
        if not math.isfinite(self.max_rel_linf) or not (0.0 < self.max_rel_linf <= 1.0):
            raise EditValidationError("max_rel_linf must be in (0.0, 1.0]")
        if not isinstance(self.protected_tensors, (tuple, list)) or not all(
            isinstance(t, str) and t for t in self.protected_tensors
        ):
            raise EditValidationError("protected_tensors must be a sequence of non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["protected_tensors"] = list(self.protected_tensors)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> EditBudget:
        if not isinstance(data, dict):
            raise EditValidationError(f"EditBudget must be a dict, got {type(data).__name__}")
        allowed_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise EditValidationError(f"Unknown keys in EditBudget: {sorted(unknown)}")
        kwargs = dict(data)
        if "protected_tensors" in kwargs:
            pt = kwargs["protected_tensors"]
            if not isinstance(pt, (list, tuple)):
                raise EditValidationError("protected_tensors must be a list or tuple of strings")
            kwargs["protected_tensors"] = tuple(pt)
        try:
            budget = cls(**kwargs)
        except TypeError as exc:
            raise EditValidationError(f"Invalid EditBudget parameters: {exc}") from exc
        budget.validate()
        return budget


@dataclass(frozen=True)
class EditOp:
    """Single declarative weight-edit operation.

    delta_scale (delta) is the relative perturbation multiplier:
      W_new = W + delta * Subspace(W).
    """

    op_type: OpType
    tensor_name: str
    delta_scale: float
    row_start: int | None = None
    row_end: int | None = None
    col_start: int | None = None
    col_end: int | None = None
    rank_start: int | None = None
    rank_end: int | None = None
    rationale: str = ""

    def validate_syntax(self, budget: EditBudget) -> None:
        if self.op_type not in VALID_OP_TYPES:
            raise EditValidationError(
                f"Unsupported op_type {self.op_type!r}. Allowed: {sorted(VALID_OP_TYPES)}"
            )
        if not self.tensor_name or not isinstance(self.tensor_name, str):
            raise EditValidationError("tensor_name must be a non-empty string")
        if (
            not budget.allow_embeddings_and_head
            and self.tensor_name in budget.protected_tensors
        ):
            raise EditValidationError(
                f"Tensor {self.tensor_name!r} is protected when allow_embeddings_and_head=False"
            )
        if not isinstance(self.delta_scale, (int, float)) or isinstance(
            self.delta_scale, bool
        ):
            raise EditValidationError("delta_scale must be a finite float")
        if not math.isfinite(float(self.delta_scale)):
            raise EditValidationError(f"Non-finite delta_scale: {self.delta_scale!r}")
        if float(self.delta_scale) == 0.0:
            raise EditValidationError("delta_scale must be non-zero")
        if abs(float(self.delta_scale)) > budget.max_abs_delta_scale:
            raise EditValidationError(
                f"|delta_scale|={abs(float(self.delta_scale)):.6f} exceeds "
                f"budget.max_abs_delta_scale={budget.max_abs_delta_scale:.6f} "
                f"on tensor {self.tensor_name!r}"
            )

        if self.op_type == "scale_tensor":
            if any(
                v is not None
                for v in (
                    self.row_start,
                    self.row_end,
                    self.col_start,
                    self.col_end,
                    self.rank_start,
                    self.rank_end,
                )
            ):
                raise EditValidationError(
                    "scale_tensor must not specify row/col/rank slice bounds"
                )
        elif self.op_type == "scale_rows":
            if self.row_start is None or self.row_end is None:
                raise EditValidationError("scale_rows requires row_start and row_end")
            if any(
                v is not None
                for v in (self.col_start, self.col_end, self.rank_start, self.rank_end)
            ):
                raise EditValidationError(
                    "scale_rows must not specify col or rank slice bounds"
                )
            if self.row_start < 0 or self.row_end <= self.row_start:
                raise EditValidationError(
                    f"Invalid row slice [{self.row_start}, {self.row_end})"
                )
        elif self.op_type == "scale_cols":
            if self.col_start is None or self.col_end is None:
                raise EditValidationError("scale_cols requires col_start and col_end")
            if any(
                v is not None
                for v in (self.row_start, self.row_end, self.rank_start, self.rank_end)
            ):
                raise EditValidationError(
                    "scale_cols must not specify row or rank slice bounds"
                )
            if self.col_start < 0 or self.col_end <= self.col_start:
                raise EditValidationError(
                    f"Invalid col slice [{self.col_start}, {self.col_end})"
                )
        elif self.op_type == "spectral_band_scale":
            if self.rank_start is None or self.rank_end is None:
                raise EditValidationError(
                    "spectral_band_scale requires rank_start and rank_end"
                )
            if any(
                v is not None
                for v in (self.row_start, self.row_end, self.col_start, self.col_end)
            ):
                raise EditValidationError(
                    "spectral_band_scale must not specify row or col slice bounds"
                )
            if self.rank_start < 0 or self.rank_end <= self.rank_start:
                raise EditValidationError(
                    f"Invalid spectral rank band [{self.rank_start}, {self.rank_end})"
                )

    def validate_against_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) not in (1, 2):
            raise EditValidationError(
                f"Tensor {self.tensor_name!r} has unsupported rank {len(shape)} (shape={shape})"
            )
        if self.op_type == "scale_rows":
            if self.row_start is None or self.row_end is None:
                raise EditValidationError("scale_rows requires row_start and row_end")
            if self.row_start < 0 or self.row_end <= self.row_start or self.row_end > shape[0]:
                raise EditValidationError(
                    f"row slice [{self.row_start}, {self.row_end}) out of bounds for "
                    f"{self.tensor_name!r} with shape {shape}"
                )
        elif self.op_type == "scale_cols":
            if len(shape) != 2:
                raise EditValidationError(
                    f"scale_cols requires a 2D tensor, got shape {shape} for {self.tensor_name!r}"
                )
            if self.col_start is None or self.col_end is None:
                raise EditValidationError("scale_cols requires col_start and col_end")
            if self.col_start < 0 or self.col_end <= self.col_start or self.col_end > shape[1]:
                raise EditValidationError(
                    f"col slice [{self.col_start}, {self.col_end}) out of bounds for "
                    f"{self.tensor_name!r} with shape {shape}"
                )
        elif self.op_type == "spectral_band_scale":
            if len(shape) != 2:
                raise EditValidationError(
                    f"spectral_band_scale requires a 2D tensor, got shape {shape} for {self.tensor_name!r}"
                )
            if self.rank_start is None or self.rank_end is None:
                raise EditValidationError(
                    "spectral_band_scale requires rank_start and rank_end"
                )
            max_rank = min(shape[0], shape[1])
            if (
                self.rank_start < 0
                or self.rank_end <= self.rank_start
                or self.rank_end > max_rank
            ):
                raise EditValidationError(
                    f"rank slice [{self.rank_start}, {self.rank_end}) exceeds min(shape)={max_rank} "
                    f"for {self.tensor_name!r} with shape {shape}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> EditOp:
        if not isinstance(data, dict):
            raise EditValidationError(f"EditOp must be a dict, got {type(data).__name__}")
        allowed_keys = {
            "op_type",
            "tensor_name",
            "delta_scale",
            "row_start",
            "row_end",
            "col_start",
            "col_end",
            "rank_start",
            "rank_end",
            "rationale",
        }
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise EditValidationError(f"Unknown keys in EditOp: {sorted(unknown)}")
        for req_key in ("op_type", "tensor_name", "delta_scale"):
            if req_key not in data:
                raise EditValidationError(f"Missing required key {req_key!r} in EditOp")
        return cls(
            op_type=data["op_type"],
            tensor_name=data["tensor_name"],
            delta_scale=float(data["delta_scale"]),
            row_start=data.get("row_start"),
            row_end=data.get("row_end"),
            col_start=data.get("col_start"),
            col_end=data.get("col_end"),
            rank_start=data.get("rank_start"),
            rank_end=data.get("rank_end"),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class EditPlan:
    """Declarative weight-edit plan proposed by an external LLM."""

    plan_id: str
    target_model: str
    hypothesis: str
    ops: list[EditOp]
    budget: EditBudget = field(default_factory=EditBudget)
    proposer_metadata: dict[str, Any] = field(default_factory=dict)

    def validate_syntax(self) -> None:
        self.budget.validate()
        if not self.plan_id or not isinstance(self.plan_id, str):
            raise EditValidationError("plan_id must be a non-empty string")
        if not self.ops:
            raise EditValidationError("EditPlan must contain at least one EditOp")
        if len(self.ops) > self.budget.max_ops:
            raise EditValidationError(
                f"EditPlan has {len(self.ops)} ops, exceeding budget.max_ops={self.budget.max_ops}"
            )
        unique_tensors = {op.tensor_name for op in self.ops}
        if len(unique_tensors) > self.budget.max_edited_tensors:
            raise EditValidationError(
                f"EditPlan targets {len(unique_tensors)} tensors, exceeding "
                f"budget.max_edited_tensors={self.budget.max_edited_tensors}"
            )
        for op in self.ops:
            op.validate_syntax(self.budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "target_model": self.target_model,
            "hypothesis": self.hypothesis,
            "budget": self.budget.to_dict(),
            "proposer_metadata": self.proposer_metadata,
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, data: Any) -> EditPlan:
        if not isinstance(data, dict):
            raise EditValidationError(f"EditPlan must be a dict, got {type(data).__name__}")
        budget_data = data.get("budget")
        budget = EditBudget.from_dict(budget_data) if budget_data is not None else EditBudget()
        ops_raw = data.get("ops")
        if not isinstance(ops_raw, list):
            raise EditValidationError("'ops' must be a list of operation dicts")
        ops = [EditOp.from_dict(item) for item in ops_raw]
        plan = cls(
            plan_id=str(data.get("plan_id", "")),
            target_model=str(data.get("target_model", "")),
            hypothesis=str(data.get("hypothesis", "")),
            ops=ops,
            budget=budget,
            proposer_metadata=dict(data.get("proposer_metadata") or {}),
        )
        plan.validate_syntax()
        return plan


@dataclass(frozen=True)
class TensorEditMetrics:
    """Pre-rounding (fp32) and post-rounding (bf16) perturbation statistics for a tensor."""

    tensor_name: str
    shape: list[int]
    numel: int
    num_ops_applied: int
    base_frobenius: float
    base_linf: float
    fp32_delta_frobenius: float
    fp32_rel_frobenius: float
    fp32_delta_linf: float
    fp32_rel_linf: float
    bf16_l0_changed_elements: int
    bf16_l0_fraction: float
    bf16_delta_frobenius: float
    bf16_rel_frobenius: float
    bf16_delta_linf: float
    bf16_rel_linf: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditExecutionReport:
    """Full audit record of an applied edit plan or matched random control."""

    plan_id: str
    mode: Literal["llm_edited", "random_control"]
    seed: int | None
    base_checkpoint_dir: str
    output_checkpoint_dir: str
    base_checkpoint_sha256: str
    output_checkpoint_sha256: str
    total_tensors_in_model: int
    num_edited_tensors: int
    total_changed_parameters_bf16: int
    global_fp32_delta_frobenius: float
    global_bf16_delta_frobenius: float
    global_max_rel_frobenius_fp32: float
    global_max_rel_linf_fp32: float
    tensor_metrics: list[TensorEditMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "mode": self.mode,
            "seed": self.seed,
            "base_checkpoint_dir": self.base_checkpoint_dir,
            "output_checkpoint_dir": self.output_checkpoint_dir,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "output_checkpoint_sha256": self.output_checkpoint_sha256,
            "total_tensors_in_model": self.total_tensors_in_model,
            "num_edited_tensors": self.num_edited_tensors,
            "total_changed_parameters_bf16": self.total_changed_parameters_bf16,
            "global_fp32_delta_frobenius": self.global_fp32_delta_frobenius,
            "global_bf16_delta_frobenius": self.global_bf16_delta_frobenius,
            "global_max_rel_frobenius_fp32": self.global_max_rel_frobenius_fp32,
            "global_max_rel_linf_fp32": self.global_max_rel_linf_fp32,
            "tensor_metrics": [m.to_dict() for m in self.tensor_metrics],
        }
