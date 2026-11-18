"""Reproducibility manifest and cryptographic checkpoint hashing.

This module uses only the Python standard library so it can be tested
independently of GPU libraries.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ConditionType = Literal["baseline", "llm_edited", "random_control"]
CHECKPOINT_EXTENSIONS = {".safetensors", ".bin", ".json", ".model", ".txt"}


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest in streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Backward-compatible string/path wrapper around ``compute_file_sha256``."""
    return compute_file_sha256(Path(path), chunk_size)


def compute_checkpoint_dir_hash(checkpoint_dir: Path) -> str:
    """Compute a deterministic, cross-platform checkpoint directory fingerprint."""
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    files = sorted(
        (
            path
            for path in checkpoint_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
        ),
        key=lambda path: path.relative_to(checkpoint_dir).as_posix(),
    )
    if not files:
        raise ValueError(f"No checkpoint files ({CHECKPOINT_EXTENSIONS}) found in {checkpoint_dir}")

    manifest_hasher = hashlib.sha256()
    for file_path in files:
        relative_path = file_path.relative_to(checkpoint_dir).as_posix()
        entry = f"{relative_path}:{compute_file_sha256(file_path)}\n".encode("utf-8")
        manifest_hasher.update(entry)
    return manifest_hasher.hexdigest()


compute_checkpoint_hash = compute_checkpoint_dir_hash


def get_git_commit_sha(repo_root: Path | None = None) -> str:
    """Return the current git commit SHA, or ``uncommitted`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
    return result.stdout.strip()


@dataclass
class EditSummary:
    """Tracks parameter count and magnitude of weight edits."""

    num_tensors_modified: int = 0
    total_params_edited_l0: int = 0
    global_l2_norm: float = 0.0
    max_abs_delta_linf: float = 0.0
    proposed_operations: list[dict[str, Any]] = field(default_factory=list)
    accepted_operations: list[dict[str, Any]] = field(default_factory=list)
    rejected_operations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EditingLLMConfig:
    """Records external LLM proposer settings."""

    model_identifier: str | None = None
    temperature: float | None = None
    prompt_template_id: str | None = None
    prompt_sha256: str | None = None
    raw_prompt_path: str | None = None
    raw_response_path: str | None = None


@dataclass
class ExperimentManifest:
    """Complete reproducibility record for one experiment condition."""

    experiment_id: str
    condition: ConditionType | str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    base_model_name: str = ""
    base_model_revision: str = ""
    base_checkpoint_path: str = ""
    base_checkpoint_sha256: str = ""
    tokenizer_revision: str = ""
    git_commit_sha: str = ""
    python_version: str = field(default_factory=platform.python_version)
    platform_info: str = field(default_factory=platform.platform)
    package_versions: dict[str, str] = field(default_factory=dict)
    hardware_device: str = ""
    precision_dtype: str = "bfloat16"
    filbench_lighteval_sha: str = "7ad5279"
    eval_backend: str = "accelerate"
    eval_config_path: str = ""
    data_split_used: str = "dev_subset"
    split_manifest_sha256: str = ""
    random_seed: int = 42
    editing_llm: EditingLLMConfig = field(default_factory=EditingLLMConfig)
    edit_summary: EditSummary = field(default_factory=EditSummary)
    output_checkpoint_path: str = ""
    output_checkpoint_sha256: str = ""
    per_task_results: dict[str, float] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    scoring_mode: str = ""
    is_partial_eval: bool = False
    is_subsampled: bool = False
    max_samples_config: int | None = None
    num_tasks_evaluated: int = 0
    evaluated_tasks: list[str] = field(default_factory=list)
    missing_tasks: list[str] = field(default_factory=list)
    subsampled_tasks: dict[str, dict[str, int]] = field(default_factory=dict)
    doc_counts_by_task: dict[str, dict[str, int]] = field(default_factory=dict)
    dataset_revisions: dict[str, str | None] = field(default_factory=dict)
    task_source_files: dict[str, str] = field(default_factory=dict)
    subtask_results: dict[str, float] = field(default_factory=dict)
    filbench_macro_score: float | None = None
    filbench_instance_weighted_score: float | None = None
    partial_macro_score_debug: float | None = None
    errors_and_observations: list[str] = field(default_factory=list)
    # Legacy fields retained for compatibility with the initial scaffold.
    model: str = ""
    seed: int | None = None
    config_sha256: str = ""
    checkpoint_sha256: str = ""
    upstream_commit: str = ""

    def validate_scientific_controls(self) -> None:
        """Enforce baseline and checkpoint non-overwrite invariants."""
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty.")
        if self.condition == "baseline":
            if self.edit_summary.total_params_edited_l0 != 0:
                raise ValueError("Baseline condition cannot have non-zero edited parameters.")
            if (
                self.base_checkpoint_sha256
                and self.output_checkpoint_sha256
                and self.base_checkpoint_sha256 != self.output_checkpoint_sha256
            ):
                raise ValueError("Baseline output_checkpoint_sha256 must match base_checkpoint_sha256.")
        elif (
            self.base_checkpoint_path
            and self.output_checkpoint_path
            and Path(self.base_checkpoint_path).resolve()
            == Path(self.output_checkpoint_path).resolve()
        ):
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: output_checkpoint_path cannot be identical "
                "to base_checkpoint_path. Never overwrite the base checkpoint!"
            )

    def save(self, path: Path) -> None:
        """Validate and write the manifest as formatted JSON."""
        self.validate_scientific_controls()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write(self, path: str | Path) -> None:
        """Backward-compatible alias for :meth:`save`."""
        self.save(Path(path))

    @classmethod
    def load(cls, path: Path) -> ExperimentManifest:
        """Load an experiment manifest from JSON."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["editing_llm"] = EditingLLMConfig(**raw.get("editing_llm", {}))
        raw["edit_summary"] = EditSummary(**raw.get("edit_summary", {}))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)
