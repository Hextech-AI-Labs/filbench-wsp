from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from filbench_wsp.proposer import (
    ingest_chatbot_response_and_materialize,
    render_zero_shot_prompt_packet,
)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render standardized editing-LLM prompt packets and ingest verbatim chatbot responses."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand 1: render-prompt
    render_p = subparsers.add_parser(
        "render-prompt",
        help="Render a deterministic Markdown Prompt Packet from the spectral profile and dev baseline.",
    )
    render_p.add_argument(
        "--packet-id",
        type=str,
        default="packet_v1_dev_zero_shot",
        help="Unique identifier for the prompt packet.",
    )
    render_p.add_argument(
        "--spectral-profile",
        type=Path,
        default=Path("configs/sailor2_1b_spectral_profile.json"),
        help="Path to sailor2_1b_spectral_profile.json.",
    )
    render_p.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("experiments/baseline_sailor2_1b_dev/score_summary.json"),
        help="Path to untouched dev baseline score_summary.json.",
    )
    render_p.add_argument(
        "--output-prompt",
        type=Path,
        default=Path("prompts/packet_v1_dev_zero_shot.md"),
        help="Output Markdown file for the prompt packet.",
    )
    render_p.add_argument(
        "--target-model",
        type=str,
        default="sail/Sailor2-1B",
        help="Target model identifier included in the prompt packet.",
    )

    # Subcommand 2: ingest-response
    ingest_p = subparsers.add_parser(
        "ingest-response",
        help="Validate a verbatim chatbot response and materialize llm_edited + random_control checkpoints.",
    )
    ingest_p.add_argument(
        "--raw-response",
        type=Path,
        required=True,
        help="Path to verbatim text file containing the chatbot response.",
    )
    ingest_p.add_argument(
        "--prompt-meta",
        type=Path,
        default=Path("prompts/packet_v1_dev_zero_shot.meta.json"),
        help="Path to the prompt packet .meta.json file.",
    )
    ingest_p.add_argument(
        "--proposer-model",
        type=str,
        required=True,
        help="Identifier of the chatbot model used (e.g., 'sonnet-5.5-high', 'gpt-6-luna-max', 'qwen-4-max').",
    )
    ingest_p.add_argument(
        "--plan-id",
        type=str,
        required=True,
        help="Canonical experiment/plan identifier (e.g., 'pilot01_sonnet55').",
    )
    ingest_p.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/base/sailor2_1b"),
        help="Path to untouched base checkpoint directory.",
    )
    ingest_p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Deterministic seed for the matched random control.",
    )
    ingest_p.add_argument(
        "--allow-tied-rows",
        action="store_true",
        help="Enable the tie-safe scale_tied_rows operator for this experiment (budget flag injected at ingest).",
    )
    ingest_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing edited/random output checkpoint directories if present.",
    )

    args = parser.parse_args()

    if args.command == "render-prompt":
        profile_path: Path = args.spectral_profile
        exp_profile = Path("experiments/sailor2_1b_spectral_profile.json")
        if not profile_path.exists() and exp_profile.exists():
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exp_profile, profile_path)
            print(f"Copied spectral profile from {exp_profile} -> {profile_path}")

        meta = render_zero_shot_prompt_packet(
            packet_id=args.packet_id,
            spectral_profile_path=profile_path,
            baseline_summary_path=args.baseline_summary,
            output_prompt_path=args.output_prompt,
            target_model=args.target_model,
        )
        print(f"Rendered Prompt Packet: {meta.prompt_markdown_path}")
        print(f"Prompt SHA-256:         {meta.prompt_sha256}")
        print(f"Metadata saved to:      {args.output_prompt.with_suffix('.meta.json')}")
        return 0

    if args.command == "ingest-response":
        plan_id: str = args.plan_id
        edited_dir = Path("checkpoints/edited") / f"{plan_id}_llm_edited"
        random_dir = Path("checkpoints/edited") / f"{plan_id}_random_control"
        bundle_dir = Path("configs/plans") / plan_id

        try:
            bundle = ingest_chatbot_response_and_materialize(
                raw_response_path=args.raw_response,
                prompt_meta_path=args.prompt_meta,
                proposer_model_id=args.proposer_model,
                base_checkpoint_dir=args.base_checkpoint,
                edited_output_dir=edited_dir,
                random_output_dir=random_dir,
                bundle_output_dir=bundle_dir,
                random_seed=args.random_seed,
                plan_id_override=plan_id,
                force=args.force,
                allow_tied_rows=args.allow_tied_rows,
            )
        except Exception as exc:
            print(
                f"[REJECTED] Plan {plan_id!r} failed validation: {exc}\n"
                f"Rejection audit saved to: {bundle_dir / 'rejection_report.json'}",
                file=sys.stderr,
            )
            return 1

        rep_e = bundle["llm_edited_report"]
        rep_r = bundle["random_control_report"]
        print(f"[ACCEPTED] Plan {plan_id!r} from proposer {args.proposer_model!r}")
        print(f"  Tied-Row Operator:         {'ENABLED' if args.allow_tied_rows else 'disabled'}")
        print(f"  Edited Tensors:            {rep_e['num_edited_tensors']}")
        print(f"  Changed Params (bf16):     {rep_e['total_changed_parameters_bf16']:,}")
        print(
            f"  Global fp32 Frobenius:     {rep_e['global_fp32_delta_frobenius']:.6f} "
            f"(random: {rep_r['global_fp32_delta_frobenius']:.6f})"
        )
        print(f"  LLM-Edited Checkpoint:     {edited_dir} (SHA256: {rep_e['output_checkpoint_sha256'][:16]}...)")
        print(f"  Random-Control Checkpoint: {random_dir} (SHA256: {rep_r['output_checkpoint_sha256'][:16]}...)")
        print(f"  Audit Bundle:              {bundle_dir / 'edit_bundle.json'}")
        return 0

    return 1

if __name__ == "__main__":
    raise SystemExit(main())