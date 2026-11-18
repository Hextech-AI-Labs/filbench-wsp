from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from filbench_wsp.editing import EditValidationError
from filbench_wsp.proposer import (
    ingest_chatbot_response_and_materialize,
    render_zero_shot_prompt_packet,
)


@pytest.fixture()
def synthetic_proposer_env(tmp_path: Path) -> dict[str, Path]:
    ckpt_dir = tmp_path / "base_ckpt"
    ckpt_dir.mkdir(parents=True)
    gen = torch.Generator(device="cpu").manual_seed(99)
    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": torch.randn((32, 32), generator=gen).to(
                torch.bfloat16
            ),
            "model.layers.0.mlp.down_proj.weight": torch.randn((32, 64), generator=gen).to(
                torch.bfloat16
            ),
        },
        str(ckpt_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "tie_word_embeddings": False}), encoding="utf-8"
    )

    profile_path = tmp_path / "spectral_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": "abc123",
                "module_type_summary": {
                    "self_attn.o_proj.weight": {
                        "count": 48,
                        "mean_frobenius_norm": 7.82,
                        "mean_linf_norm": 0.17,
                        "mean_condition_number": 19259.6,
                        "mean_effective_rank": 624.8,
                        "mean_energy_rank_80": 244.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "score_summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "overall_score": 31.75,
                "category_scores": {"Classical NLP": 35.43, "Generation": 16.97},
                "task_scores": {"balita_tgl_mcf": 34.0, "ntrex128_fil": 15.4},
            }
        ),
        encoding="utf-8",
    )

    return {
        "ckpt_dir": ckpt_dir,
        "profile_path": profile_path,
        "baseline_path": baseline_path,
    }


def test_render_prompt_and_ingest_valid_and_rejected_responses(
    synthetic_proposer_env: dict[str, Path], tmp_path: Path
) -> None:
    prompt_path = tmp_path / "prompts" / "packet_test.md"
    meta = render_zero_shot_prompt_packet(
        packet_id="packet_test",
        spectral_profile_path=synthetic_proposer_env["profile_path"],
        baseline_summary_path=synthetic_proposer_env["baseline_path"],
        output_prompt_path=prompt_path,
    )
    assert prompt_path.exists()
    assert len(meta.prompt_sha256) == 64

    # 1. Valid chatbot response inside markdown fence with commentary
    valid_raw = tmp_path / "valid_response.txt"
    valid_raw.write_text(
        """Here is my proposed weight edit based on the spectral profile:
```json
{
  "plan_id": "chatbot_plan",
  "target_model": "sail/Sailor2-1B",
  "hypothesis": "Attenuate top singular components of layer 0 o_proj",
  "ops": [
    {
      "op_type": "spectral_band_scale",
      "tensor_name": "model.layers.0.self_attn.o_proj.weight",
      "delta_scale": -0.06,
      "rank_start": 0,
      "rank_end": 4,
      "rationale": "Dampen dominant routing anisotropy"
    }
  ]
}
```
Hope this helps!""",
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "bundle_ok"
    bundle = ingest_chatbot_response_and_materialize(
        raw_response_path=valid_raw,
        prompt_meta_path=prompt_path.with_suffix(".meta.json"),
        proposer_model_id="sonnet-5.5-high",
        base_checkpoint_dir=synthetic_proposer_env["ckpt_dir"],
        edited_output_dir=tmp_path / "edited_ckpt",
        random_output_dir=tmp_path / "random_ckpt",
        bundle_output_dir=bundle_dir,
        random_seed=42,
        plan_id_override="pilot01_sonnet55",
    )
    assert bundle["status"] == "accepted"
    assert bundle["plan"]["proposer_metadata"]["proposer_model_id"] == "sonnet-5.5-high"
    assert (bundle_dir / "edit_bundle.json").exists()

    # 2. Invalid chatbot response exceeding budget must fail closed and write rejection_report.json
    invalid_raw = tmp_path / "invalid_response.txt"
    invalid_raw.write_text(
        """```json
{
  "plan_id": "bad_plan",
  "target_model": "sail/Sailor2-1B",
  "hypothesis": "Too large scale",
  "ops": [
    {
      "op_type": "scale_tensor",
      "tensor_name": "model.layers.0.self_attn.o_proj.weight",
      "delta_scale": 0.90
    }
  ]
}
```""",
        encoding="utf-8",
    )
    reject_dir = tmp_path / "bundle_reject"
    with pytest.raises(EditValidationError):
        ingest_chatbot_response_and_materialize(
            raw_response_path=invalid_raw,
            prompt_meta_path=prompt_path.with_suffix(".meta.json"),
            proposer_model_id="gpt-6-luna-max",
            base_checkpoint_dir=synthetic_proposer_env["ckpt_dir"],
            edited_output_dir=tmp_path / "edited_bad",
            random_output_dir=tmp_path / "random_bad",
            bundle_output_dir=reject_dir,
        )
    assert (reject_dir / "rejection_report.json").exists()
