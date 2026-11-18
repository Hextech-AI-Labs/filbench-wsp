from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filbench_wsp.editing import (
    EditBudget,
    EditPlan,
    EditValidationError,
    apply_edit_plan_to_checkpoint,
)


@dataclass(frozen=True)
class PromptPacketMetadata:
    packet_id: str
    created_at_utc: str
    target_model: str
    base_checkpoint_sha256: str
    spectral_profile_path: str
    spectral_profile_sha256: str
    baseline_summary_path: str
    baseline_summary_sha256: str
    prompt_markdown_path: str
    prompt_sha256: str
    budget: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_layer_index(tensor_name: str) -> int | None:
    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", tensor_name)
    return int(m.group(1)) if m else None


def _summarize_spectral_profile_for_prompt(profile: dict[str, Any]) -> str:
    """Format module-level and depth-band spectral statistics into compact Markdown tables."""
    lines: list[str] = []

    # 1. Module-type summary table
    mod_summary = (
        profile.get("module_type_averages")
        or profile.get("module_type_summary")
        or profile.get("module_summaries")
        or profile.get("summary_by_module_type")
        or {}
    )
    if isinstance(mod_summary, dict) and mod_summary:
        num_layers = profile.get("num_hidden_layers")
        layer_scope = (
            f"Across All {int(num_layers)} Layers"
            if isinstance(num_layers, int)
            else "Across All Layers"
        )
        lines.append(f"### Module-Type Spectral Averages ({layer_scope})")
        lines.append(
            "| Module Suffix | Count | Mean Frobenius | Mean L_inf | Mean Cond (kappa) | Mean Effective Rank | Mean R_80% |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for mod_name, stats in sorted(mod_summary.items()):
            if not isinstance(stats, dict):
                continue
            count = stats.get("count", "")
            frob = stats.get(
                "avg_frobenius",
                stats.get("mean_frobenius_norm", stats.get("frobenius_norm", 0.0)),
            )
            linf = stats.get(
                "avg_linf",
                stats.get("mean_linf_norm", stats.get("linf_norm", 0.0)),
            )
            cond = stats.get(
                "avg_condition_number",
                stats.get("mean_condition_number", stats.get("condition_number", "N/A")),
            )
            reff = stats.get(
                "avg_effective_rank",
                stats.get("mean_effective_rank", stats.get("effective_rank", "N/A")),
            )
            r80 = stats.get(
                "avg_energy_rank_80pct",
                stats.get(
                    "mean_energy_rank_80",
                    stats.get("energy_rank_80", stats.get("energy_rank_80pct", "N/A")),
                ),
            )

            cond_str = f"{float(cond):.2f}" if isinstance(cond, (int, float)) else str(cond)
            reff_str = f"{float(reff):.2f}" if isinstance(reff, (int, float)) else str(reff)
            r80_str = f"{float(r80):.1f}" if isinstance(r80, (int, float)) else str(r80)

            lines.append(
                f"| `{mod_name}` | {count} | {float(frob):.4f} | {float(linf):.4f} | "
                f"{cond_str} | {reff_str} | {r80_str} |"
            )
        lines.append("")

    # 2. Per-layer / depth-sampled tensor table (sample key layers across model depth)
    raw_tensors = (
        profile.get("tensor_profiles")
        or profile.get("tensors")
        or profile.get("layer_profiles")
        or []
    )
    tensor_list: list[dict[str, Any]] = []
    if isinstance(raw_tensors, dict):
        for k, v in raw_tensors.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("name", k)
                tensor_list.append(item)
    elif isinstance(raw_tensors, list):
        tensor_list = [t for t in raw_tensors if isinstance(t, dict)]

    if tensor_list:
        sampled_layers = {0, 1, 7, 15, 23, 31, 39, 46, 47}
        lines.append(
            "### Depth-Sampled Layer Spectral Profiles (Layers 0, 1, 7, 15, 23, 31, 39, 46, 47)"
        )
        lines.append(
            "*(All layers 0..47 exist in the checkpoint with identical shapes; intermediate layers interpolate smoothly.)*"
        )
        lines.append(
            "| Tensor Name | Shape | Frobenius | L_inf | Effective Rank | R_80% |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for t in tensor_list:
            tname = str(t.get("name") or t.get("tensor_name") or "")
            layer_idx = _extract_layer_index(tname)
            if layer_idx is not None and layer_idx not in sampled_layers:
                continue
            if tname in ("model.embed_tokens.weight", "lm_head.weight"):
                continue
            shape = t.get("shape", [])
            if isinstance(shape, list) and len(shape) < 2 and tname != "model.norm.weight":
                continue
            frob = float(t.get("frobenius_norm", 0.0))
            linf = float(t.get("linf_norm", 0.0))
            reff = t.get("effective_rank")
            r80 = t.get("energy_rank_80pct", t.get("energy_rank_80"))
            reff_str = f"{float(reff):.1f}" if reff is not None else "N/A"
            r80_str = f"{float(r80):.0f}" if r80 is not None else "N/A"
            lines.append(
                f"| `{tname}` | `{shape}` | {frob:.4f} | {linf:.4f} | {reff_str} | {r80_str} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_zero_shot_prompt_packet(
    packet_id: str,
    spectral_profile_path: Path,
    baseline_summary_path: Path,
    output_prompt_path: Path,
    budget: EditBudget | None = None,
    target_model: str = "sail/Sailor2-1B",
) -> PromptPacketMetadata:
    """Render a deterministic, self-contained Markdown Prompt Packet for a chatbot editing LLM."""
    active_budget = budget or EditBudget()
    active_budget.validate()

    profile = json.loads(spectral_profile_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    base_sha = str(
        profile.get("checkpoint_sha256")
        or profile.get("base_checkpoint_sha256")
        or baseline.get("base_checkpoint_sha256")
        or "095c21b1a7c3bc6d8959090e580fbda7ef30d3bc2d3960f0eed66f9ffa8aa6c6"
    )

    cat_scores = baseline.get("category_scores") or {}
    task_scores = baseline.get("per_task_scores_pct") or baseline.get("task_scores") or {}
    overall_score = next(
        (
            baseline.get(key)
            for key in (
                "filbench_macro_score",
                "overall_score",
                "partial_macro_score_debug",
                "unweighted_partial_macro_score",
            )
            if baseline.get(key) is not None
        ),
        None,
    )

    model_type = str(profile.get("model_type") or "unknown")
    model_class = {
        "qwen3": "Qwen3ForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
    }.get(model_type, model_type)
    num_layers = int(profile.get("num_hidden_layers", 0))
    hidden_size = int(profile.get("hidden_size", 0))
    num_attention_heads = int(profile.get("num_attention_heads", 0))
    num_key_value_heads = int(profile.get("num_key_value_heads", 0))
    head_dim = int(profile.get("head_dim", 0))
    editable_names = profile.get("editable_tensor_names") or []
    tensor_profiles = {
        item["name"]: item
        for item in profile.get("tensor_profiles", [])
        if isinstance(item, dict) and item.get("name")
    }
    editable_suffixes = sorted(
        {
            name.removeprefix("model.layers.0.")
            for name in editable_names
            if name.startswith("model.layers.0.")
        }
    )
    editable_lines = [
        f"  - `model.layers.{{L}}.{suffix}` "
        f"({tensor_profiles.get(f'model.layers.0.{suffix}', {}).get('shape', 'shape from profile')})"
        for suffix in editable_suffixes
    ]
    global_editable = [
        name for name in editable_names if not name.startswith("model.layers.")
    ]
    protected_names = profile.get("protected_tensor_names") or []
    architecture_section = "\n".join(
        [
            f"- **Architecture:** `{model_class}` (`{profile.get('dtype', 'bfloat16')}`)",
            f"- **Depth & Width:** `{num_layers}` transformer layers "
            f"(`model.layers.0` .. `model.layers.{max(num_layers - 1, 0)}`), "
            f"`hidden_size = {hidden_size}`",
            f"- **Attention:** `{num_attention_heads}` query heads, "
            f"`{num_key_value_heads}` key/value heads, `head_dim = {head_dim}`",
            "- **Per-Layer Editable Tensors:**",
            *editable_lines,
            *[f"- **Global Editable Tensor:** `{name}`" for name in global_editable],
            f"- **Protected Tensors (DO NOT EDIT):** "
            f"{', '.join(f'`{name}`' for name in protected_names)}",
        ]
    )
    overall_score_text = (
        f"{float(overall_score):.2f}%"
        if overall_score is not None
        else "not available for this partial/dev scoring mode"
    )

    spectral_md = _summarize_spectral_profile_for_prompt(profile)

    task_lines = [
        "| Benchmark Task | Score (%) |",
        "| :--- | :---: |",
    ]
    for task_name, val in sorted(task_scores.items()):
        score_val = val.get("score", val) if isinstance(val, dict) else val
        task_lines.append(f"| `{task_name}` | {float(score_val):.2f} |")
    tasks_md = "\n".join(task_lines)

    cat_lines = [
        "| FilBench Category | Dev Macro Score (%) |",
        "| :--- | :---: |",
    ]
    for cat_name, val in sorted(cat_scores.items()):
        cat_lines.append(f"| {cat_name} | {float(val):.2f} |")
    cats_md = "\n".join(cat_lines)

    prompt_text = f"""# Direct Weight-Space Intervention Task (FilBench-WSP)
Packet ID: `{packet_id}`

You are participating in a scientific experiment testing whether a frontier LLM can propose a **direct, training-free weight perturbation** ($\\theta_{{\\text{{edited}}}} = \\theta_{{\\text{{base}}}} + \\Delta\\theta$) to improve the Filipino/Tagalog/Cebuano performance of `{target_model}` on the FilBench development split, without gradient descent, fine-tuning, or adapters.

---

## 1. Target Model Architecture (`{target_model}`)
{architecture_section}

---

## 2. Structural & Spectral Profile of Untouched Checkpoint
{spectral_md}

---

## 3. Untouched Development Baseline Scores
- **Overall Unweighted Category Macro Score:** `{overall_score_text}`

{cats_md}

### Per-Benchmark Dev Breakdown
{tasks_md}

---

## 4. Allowed Weight-Edit Operator Algebra & Safety Budget

Every operation in `ops` computes a perturbation $\\Delta W$ in `float32` using relative multiplier `delta_scale` ($\\delta$), applied as $W_{{\\text{{new}}}} = \\text{{bf16}}(W + \\Delta W)$:

1. `"scale_tensor"`: Whole-tensor scaling $\\Delta W = \\delta W$.
   - Must NOT set `row_start`, `row_end`, `col_start`, `col_end`, `rank_start`, or `rank_end`.
2. `"scale_rows"`: Contiguous row-slice scaling $\\Delta W_{{r_0:r_1, \\dots}} = \\delta W_{{r_0:r_1, \\dots}}$.
   - Requires integer `row_start` ($r_0$) and `row_end` ($r_1$) with $0 \\le r_0 < r_1 \\le \\text{{shape}}[0]$.
3. `"scale_cols"`: Contiguous column-slice scaling on 2D matrices $\\Delta W_{{: , c_0:c_1}} = \\delta W_{{: , c_0:c_1}}$.
   - Requires integer `col_start` ($c_0$) and `col_end` ($c_1$) with $0 \\le c_0 < c_1 \\le \\text{{shape}}[1]$.
4. `"spectral_band_scale"`: Truncated SVD singular-band scaling on 2D matrices:
   $$\\Delta W = \\sum_{{i=k_0}}^{{k_1 - 1}} \\delta \\, \\sigma_i \\, u_i v_i^\\top$$
   - Requires integer `rank_start` ($k_0$) and `rank_end` ($k_1$) with $0 \\le k_0 < k_1 \\le \\min(\\text{{shape}}[0], \\text{{shape}}[1])$.

### Fail-Closed Safety Budget (Strict Enforcement)
- `max_ops`: `{active_budget.max_ops}`
- `max_edited_tensors`: `{active_budget.max_edited_tensors}`
- Per-op `|delta_scale|`: `0.01 <= |delta_scale| <= {active_budget.max_abs_delta_scale}` (Note: `|delta_scale| >= 0.01` is required so the edit does not underflow to zero in `bfloat16`!)
- Per-tensor relative Frobenius norm: $\\|\\Delta W\\|_F / \\|W\\|_F \\le {active_budget.max_rel_frobenius}$
- Per-tensor relative max-entry norm: $\\|\\Delta W\\|_\\infty / \\|W\\|_\\infty \\le {active_budget.max_rel_linf}$

---

## 5. Required Output Format

Return **only** a single valid JSON object inside a ```json ... ``` code block conforming to this exact schema:

```json
{{
  "plan_id": "<short_snake_case_identifier>",
  "target_model": "{target_model}",
  "hypothesis": "<concise mechanistic hypothesis explaining why these specific layers/subspaces and delta_scale signs should improve Filipino/Cebuano MCF and generation performance without collapsing fluency>",
  "ops": [
    {{
      "op_type": "spectral_band_scale",
      "tensor_name": "model.layers.40.self_attn.o_proj.weight",
      "delta_scale": -0.05,
      "rank_start": 0,
      "rank_end": 8,
      "rationale": "<why this specific tensor, band/slice, and sign were chosen>"
    }}
  ]
}}
```
"""

    output_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    output_prompt_path.write_text(prompt_text, encoding="utf-8")

    meta = PromptPacketMetadata(
        packet_id=packet_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        target_model=target_model,
        base_checkpoint_sha256=base_sha,
        spectral_profile_path=str(spectral_profile_path),
        spectral_profile_sha256=_sha256_file(spectral_profile_path),
        baseline_summary_path=str(baseline_summary_path),
        baseline_summary_sha256=_sha256_file(baseline_summary_path),
        prompt_markdown_path=str(output_prompt_path),
        prompt_sha256=_sha256_text(prompt_text),
        budget=active_budget.to_dict(),
    )
    meta_path = output_prompt_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return meta


def extract_json_from_raw_response(raw_text: str) -> dict[str, Any]:
    """Extract a single JSON object from a verbatim chatbot response without altering values."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates = list(fenced) if fenced else []

    # Also search for code blocks that might have commentary inside
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.DOTALL)
    for block in fenced_blocks:
        s = block.find("{")
        e = block.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidates.append(block[s : e + 1])

    # Outermost curly braces in raw text
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    if not candidates:
        raise EditValidationError("No JSON object found in raw chatbot response")

    last_err: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "ops" in parsed:
                return parsed
        except Exception as exc:
            last_err = exc

    raise EditValidationError(
        f"Failed to parse valid EditPlan JSON object from chatbot response: {last_err}"
    )


def ingest_chatbot_response_and_materialize(
    raw_response_path: Path,
    prompt_meta_path: Path,
    proposer_model_id: str,
    base_checkpoint_dir: Path,
    edited_output_dir: Path,
    random_output_dir: Path,
    bundle_output_dir: Path,
    random_seed: int = 42,
    plan_id_override: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Parse a verbatim chatbot response, validate fail-closed, and materialize edited + random checkpoints."""
    raw_text = raw_response_path.read_text(encoding="utf-8")
    raw_sha256 = _sha256_text(raw_text)
    prompt_meta = json.loads(prompt_meta_path.read_text(encoding="utf-8"))

    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "proposer_model_id": proposer_model_id,
        "interface_mode": "chat_ui_zero_history_packet",
        "packet_id": prompt_meta.get("packet_id"),
        "prompt_sha256": prompt_meta.get("prompt_sha256"),
        "raw_response_path": str(raw_response_path),
        "raw_response_sha256": raw_sha256,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_control_seed": random_seed,
    }

    try:
        parsed_dict = extract_json_from_raw_response(raw_text)
        if plan_id_override:
            parsed_dict["plan_id"] = plan_id_override
        parsed_dict["proposer_metadata"] = {
            **dict(parsed_dict.get("proposer_metadata") or {}),
            **provenance,
            "proposed_ops_count": len(parsed_dict.get("ops", []))
            if isinstance(parsed_dict.get("ops"), list)
            else 0,
        }
        plan = EditPlan.from_dict(parsed_dict)

        report_edited = apply_edit_plan_to_checkpoint(
            plan=plan,
            base_checkpoint_dir=base_checkpoint_dir,
            output_checkpoint_dir=edited_output_dir,
            mode="llm_edited",
            force=force,
        )
        report_random = apply_edit_plan_to_checkpoint(
            plan=plan,
            base_checkpoint_dir=base_checkpoint_dir,
            output_checkpoint_dir=random_output_dir,
            mode="random_control",
            random_seed=random_seed,
            force=force,
        )
    except Exception as exc:
        rejection_record = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "provenance": provenance,
        }
        (bundle_output_dir / "rejection_report.json").write_text(
            json.dumps(rejection_record, indent=2), encoding="utf-8"
        )
        raise

    # Save audit artifacts in bundle_output_dir and inside each materialized checkpoint
    plan_dict = plan.to_dict()
    plan_dict["proposer_metadata"]["accepted_ops_count"] = len(plan.ops)

    bundle = {
        "status": "accepted",
        "plan": plan_dict,
        "llm_edited_report": report_edited.to_dict(),
        "random_control_report": report_random.to_dict(),
    }
    (bundle_output_dir / "edit_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )
    (edited_output_dir / "filbench_wsp_edit_report.json").write_text(
        json.dumps({"plan": plan_dict, "execution_report": report_edited.to_dict()}, indent=2),
        encoding="utf-8",
    )
    (random_output_dir / "filbench_wsp_edit_report.json").write_text(
        json.dumps({"plan": plan_dict, "execution_report": report_random.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return bundle
