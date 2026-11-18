"""Model architecture, VRAM footprint, and Filipino tokenizer fertility inspector.

Used in Phase 1 to compare candidate base models before downloading full
checkpoints, and in Phase 2 to extract structural and spectral weight profiles.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filbench_wsp.editing.schema import DEFAULT_PROTECTED_TENSORS
from filbench_wsp.manifest import compute_checkpoint_hash

BYTES_PER_DTYPE: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for hardware inspection. "
            "Install the project with its runtime dependencies or run: "
            "pip install torch"
        ) from exc
    return torch


@dataclass(frozen=True)
class HardwareStatus:
    cuda_available: bool
    device_name: str
    total_vram_gb: float
    bf16_supported: bool


@dataclass(frozen=True)
class MemoryEstimate:
    model_id: str
    dtype: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    weights_gb: float
    kv_cache_gb: float
    estimated_total_inference_gb: float
    fits_in_6gb_vram: bool


@dataclass(frozen=True)
class FertilityReport:
    model_id: str
    vocab_size: int
    total_words: int
    total_tokens: int
    fertility_tokens_per_word: float
    chars_per_token: float
    sample_tokenizations: dict[str, list[str]]


@dataclass(frozen=True)
class TensorSpectralSummary:
    """Structural, norm, and spectral properties of a single tensor."""

    name: str
    shape: list[int]
    dtype: str
    numel: int
    frobenius_norm: float
    linf_norm: float
    mean: float
    std: float
    min_val: float
    max_val: float
    is_2d: bool
    max_rank: int | None = None
    spectral_norm: float | None = None  # sigma_1
    min_singular_value: float | None = None  # sigma_min
    condition_number: float | None = None  # sigma_max / sigma_min
    effective_rank: float | None = None  # Roy & Vetterli entropy effective rank
    top_5_singular_values: list[float] | None = None
    energy_rank_50pct: int | None = None
    energy_rank_80pct: int | None = None
    energy_rank_90pct: int | None = None
    energy_rank_95pct: int | None = None
    band_energy_fractions: dict[str, float] | None = None  # Q1, Q2, Q3, Q4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointStructuralProfile:
    """Comprehensive structural and spectral inventory of a model checkpoint."""

    checkpoint_dir: str
    checkpoint_sha256: str
    model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    total_parameters: int
    total_tensors: int
    protected_tensor_names: list[str]
    editable_tensor_names: list[str]
    module_type_averages: dict[str, dict[str, float]]
    tensor_profiles: list[TensorSpectralSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_hardware_status() -> HardwareStatus:
    """Inspect local GPU availability and bfloat16 support."""
    try:
        torch = _load_torch()
    except ImportError:
        return HardwareStatus(
            cuda_available=False,
            device_name="CPU",
            total_vram_gb=0.0,
            bf16_supported=False,
        )
    if not torch.cuda.is_available():
        return HardwareStatus(
            cuda_available=False,
            device_name="CPU",
            total_vram_gb=0.0,
            bf16_supported=False,
        )
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    return HardwareStatus(
        cuda_available=True,
        device_name=props.name,
        total_vram_gb=round(total_gb, 2),
        bf16_supported=torch.cuda.is_bf16_supported(),
    )


def available_vram_bytes() -> int | None:
    """Return visible NVIDIA VRAM when available, otherwise ``None``."""
    try:
        torch = _load_torch()
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).total_memory)


def disk_free_bytes(path: str = ".") -> int:
    """Return free bytes for the filesystem containing ``path``."""
    return shutil.disk_usage(path).free


def estimate_vram_from_config(
    model_id: str,
    approx_param_count: int,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    seq_len: int = 4096,
    cuda_overhead_gb: float = 0.6,
) -> MemoryEstimate:
    """Estimate weights and GQA KV-cache memory from a Hugging Face config."""
    try:
        bytes_per_elem = BYTES_PER_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc
    if approx_param_count < 0 or batch_size < 1 or seq_len < 1:
        raise ValueError("Parameter count, batch size, and sequence length must be positive.")

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "Transformers is required for model configuration inspection. "
            "Install the project with its runtime dependencies."
        ) from exc
    cfg = AutoConfig.from_pretrained(model_id)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    hidden_size = int(getattr(cfg, "hidden_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // num_heads))
    vocab_size = int(getattr(cfg, "vocab_size"))

    weights_bytes = approx_param_count * bytes_per_elem
    weights_gb = weights_bytes / (1024**3)
    kv_cache_bytes = (
        2
        * batch_size
        * seq_len
        * num_layers
        * num_kv_heads
        * head_dim
        * bytes_per_elem
    )
    kv_cache_gb = kv_cache_bytes / (1024**3)
    total_gb = weights_gb + kv_cache_gb + cuda_overhead_gb

    return MemoryEstimate(
        model_id=model_id,
        dtype=dtype,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        weights_gb=round(weights_gb, 3),
        kv_cache_gb=round(kv_cache_gb, 3),
        estimated_total_inference_gb=round(total_gb, 3),
        fits_in_6gb_vram=(total_gb < 5.5),
    )


def compute_filipino_fertility(
    model_id: str,
    sentences: list[str],
    probe_words: list[str],
) -> FertilityReport:
    """Measure tokenizer subword fertility on Filipino-language sentences."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformers is required for tokenizer inspection. "
            "Install the project with its runtime dependencies."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    words: list[str] = []
    for sentence in sentences:
        words.extend(sentence.strip().split())
    if not words:
        raise ValueError("Sentences list must contain at least one word.")

    token_counts = [len(tokenizer.encode(word, add_special_tokens=False)) for word in words]
    total_words = len(words)
    total_tokens = sum(token_counts)
    total_chars = sum(len(word) for word in words)
    fertility = total_tokens / total_words
    chars_per_token = total_chars / max(total_tokens, 1)

    sample_map: dict[str, list[str]] = {}
    for word in probe_words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        sample_map[word] = [tokenizer.decode([token_id]) for token_id in token_ids]

    return FertilityReport(
        model_id=model_id,
        vocab_size=len(tokenizer),
        total_words=total_words,
        total_tokens=total_tokens,
        fertility_tokens_per_word=round(fertility, 3),
        chars_per_token=round(chars_per_token, 3),
        sample_tokenizations=sample_map,
    )


def compute_tensor_spectral_summary(name: str, tensor: Any) -> TensorSpectralSummary:
    """Compute Frobenius norm, L_inf norm, and SVD spectral properties of a tensor."""
    torch = _load_torch()
    shape = list(tensor.shape)
    numel = int(tensor.numel())
    dtype_str = str(tensor.dtype).replace("torch.", "")

    t_fp32 = tensor.to(torch.float32)
    frob = float(torch.linalg.vector_norm(t_fp32).item())
    linf = float(torch.max(torch.abs(t_fp32)).item())
    mean_val = float(t_fp32.mean().item())
    std_val = float(t_fp32.std().item()) if numel > 1 else 0.0
    min_val = float(t_fp32.min().item())
    max_val = float(t_fp32.max().item())

    is_2d = len(shape) == 2
    if not is_2d:
        return TensorSpectralSummary(
            name=name,
            shape=shape,
            dtype=dtype_str,
            numel=numel,
            frobenius_norm=round(frob, 6),
            linf_norm=round(linf, 6),
            mean=round(mean_val, 6),
            std=round(std_val, 6),
            min_val=round(min_val, 6),
            max_val=round(max_val, 6),
            is_2d=False,
        )

    # 2D weight matrix: compute SVD singular values
    s = torch.linalg.svdvals(t_fp32)
    max_rank = min(shape[0], shape[1])
    s_max = float(s[0].item())
    s_min = float(s[-1].item())
    cond = s_max / max(s_min, 1e-12)

    # Roy & Vetterli effective rank = exp(entropy(normalized singular values))
    s_sum = float(s.sum().item())
    if s_sum > 1e-12:
        p = s / s_sum
        entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
        erank = float(math.exp(entropy))
    else:
        erank = 0.0

    # Cumulative Frobenius energy: sum(sigma_i^2) / total_energy
    total_energy = float((s**2).sum().item())
    cum_energy = (torch.cumsum(s**2, dim=0) / max(total_energy, 1e-12)).cpu()

    def _rank_for_threshold(th: float) -> int:
        idx = int(torch.searchsorted(cum_energy, th).item())
        return min(idx + 1, max_rank)

    r50 = _rank_for_threshold(0.50)
    r80 = _rank_for_threshold(0.80)
    r90 = _rank_for_threshold(0.90)
    r95 = _rank_for_threshold(0.95)

    # Quartile band energy distribution
    q1_cut = max_rank // 4
    q2_cut = max_rank // 2
    q3_cut = (3 * max_rank) // 4
    e_q1 = float((s[:q1_cut] ** 2).sum().item()) / max(total_energy, 1e-12)
    e_q2 = float((s[q1_cut:q2_cut] ** 2).sum().item()) / max(total_energy, 1e-12)
    e_q3 = float((s[q2_cut:q3_cut] ** 2).sum().item()) / max(total_energy, 1e-12)
    e_q4 = float((s[q3_cut:] ** 2).sum().item()) / max(total_energy, 1e-12)

    top_5 = [round(float(val), 6) for val in s[:5].tolist()]

    return TensorSpectralSummary(
        name=name,
        shape=shape,
        dtype=dtype_str,
        numel=numel,
        frobenius_norm=round(frob, 6),
        linf_norm=round(linf, 6),
        mean=round(mean_val, 6),
        std=round(std_val, 6),
        min_val=round(min_val, 6),
        max_val=round(max_val, 6),
        is_2d=True,
        max_rank=max_rank,
        spectral_norm=round(s_max, 6),
        min_singular_value=round(s_min, 6),
        condition_number=round(cond, 3),
        effective_rank=round(erank, 2),
        top_5_singular_values=top_5,
        energy_rank_50pct=r50,
        energy_rank_80pct=r80,
        energy_rank_90pct=r90,
        energy_rank_95pct=r95,
        band_energy_fractions={
            "Q1_0_25pct": round(e_q1, 4),
            "Q2_25_50pct": round(e_q2, 4),
            "Q3_50_75pct": round(e_q3, 4),
            "Q4_75_100pct": round(e_q4, 4),
        },
    )


def _classify_module_type(tensor_name: str) -> str:
    """Group parameter names into canonical architectural module types."""
    for pattern in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "embed_tokens",
        "lm_head",
    ):
        if pattern in tensor_name:
            suffix = "bias" if tensor_name.endswith(".bias") else "weight"
            return f"{pattern}.{suffix}"
    return "other"


def inspect_checkpoint_spectral_profile(
    checkpoint_dir: Path,
    protected_tensors: tuple[str, ...] = DEFAULT_PROTECTED_TENSORS,
) -> CheckpointStructuralProfile:
    """Inspect all safetensors files in a checkpoint and produce full spectral profiles."""
    from safetensors import safe_open

    ckpt_path = checkpoint_dir.resolve()
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    # Load configuration
    cfg_data: dict[str, Any] = {}
    config_file = ckpt_path / "config.json"
    if config_file.exists():
        try:
            cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            cfg_data = {}

    st_files = sorted(ckpt_path.glob("*.safetensors"))
    if not st_files:
        raise ValueError(f"No .safetensors files found in {ckpt_path}")

    checkpoint_sha = compute_checkpoint_hash(ckpt_path)

    tensor_profiles: list[TensorSpectralSummary] = []
    total_params = 0

    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                tensor = handle.get_tensor(key)
                summary = compute_tensor_spectral_summary(key, tensor)
                tensor_profiles.append(summary)
                total_params += summary.numel

    # Group averages by module type
    by_type: dict[str, list[TensorSpectralSummary]] = defaultdict(list)
    for p in tensor_profiles:
        mod_type = _classify_module_type(p.name)
        by_type[mod_type].append(p)

    module_averages: dict[str, dict[str, float]] = {}
    for mod_type, summaries in sorted(by_type.items()):
        cnt = len(summaries)
        avg_frob = sum(s.frobenius_norm for s in summaries) / cnt
        avg_linf = sum(s.linf_norm for s in summaries) / cnt
        row: dict[str, float] = {
            "count": cnt,
            "avg_frobenius": round(avg_frob, 4),
            "avg_linf": round(avg_linf, 4),
        }
        summaries_2d = [s for s in summaries if s.is_2d and s.condition_number is not None]
        if summaries_2d:
            c2d = len(summaries_2d)
            row["avg_condition_number"] = round(
                sum(s.condition_number for s in summaries_2d) / c2d, 2
            )
            row["avg_effective_rank"] = round(
                sum(s.effective_rank for s in summaries_2d if s.effective_rank is not None) / c2d,
                2,
            )
            row["avg_energy_rank_80pct"] = round(
                sum(s.energy_rank_80pct for s in summaries_2d if s.energy_rank_80pct is not None)
                / c2d,
                1,
            )
        module_averages[mod_type] = row

    protected_set = set(protected_tensors)
    protected_names = [p.name for p in tensor_profiles if p.name in protected_set]
    editable_names = [p.name for p in tensor_profiles if p.name not in protected_set]

    num_layers = int(cfg_data.get("num_hidden_layers", 0))
    hidden_size = int(cfg_data.get("hidden_size", 0))
    num_heads = int(cfg_data.get("num_attention_heads", 0))
    num_kv_heads = int(cfg_data.get("num_key_value_heads", num_heads))
    head_dim = int(cfg_data.get("head_dim", (hidden_size // num_heads) if num_heads else 0))
    vocab_size = int(cfg_data.get("vocab_size", 0))
    tie_embeddings = bool(cfg_data.get("tie_word_embeddings", False))
    model_type = str(cfg_data.get("model_type", "unknown"))

    return CheckpointStructuralProfile(
        checkpoint_dir=str(ckpt_path),
        checkpoint_sha256=checkpoint_sha,
        model_type=model_type,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_embeddings,
        total_parameters=total_params,
        total_tensors=len(tensor_profiles),
        protected_tensor_names=protected_names,
        editable_tensor_names=editable_names,
        module_type_averages=module_averages,
        tensor_profiles=tensor_profiles,
    )


def to_dict(report: Any) -> dict[str, Any]:
    """Serialize any inspection dataclass to a JSON-friendly dictionary."""
    return asdict(report)
