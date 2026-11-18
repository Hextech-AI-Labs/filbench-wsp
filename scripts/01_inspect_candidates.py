"""Inspect model checkpoints for structural/spectral profiles and compare candidate models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from filbench_wsp.model_inspect import (
    compute_filipino_fertility,
    estimate_vram_from_config,
    get_hardware_status,
    inspect_checkpoint_spectral_profile,
    to_dict,
)

SAMPLE_FILIPINO_SENTENCES = [
    "Ang pananaliksik na ito ay sumusuri sa kakayahan ng modelo sa wikang Filipino.",
    "Kailangan nating tiyakin na tama at patas ang pagsusuri bago gumawa ng konklusyon.",
    "Maayong buntag sa tanan nga nagtuon sa pinulongang Binisaya ug Sinugboanon.",
    "Nagpadayon ang pag-uswag sa teknolohiya alang sa mga lokal na pinulongan sa Pilipinas.",
]

PROBE_WORDS = [
    "nakakapagpabagabag",
    "pananaliksik",
    "pagkakaisa",
    "pinagsamahan",
    "kaugalingon",
    "sinugboanon",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint spectral profiles and candidate model capabilities."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/base/sailor2_1b",
        help="Path to local checkpoint directory to inspect (default: checkpoints/base/sailor2_1b).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/sailor2_1b_spectral_profile.json",
        help="Path to save the spectral profile JSON (default: experiments/sailor2_1b_spectral_profile.json).",
    )
    parser.add_argument(
        "--compare-candidates",
        action="store_true",
        help="Also compare remote candidate models from configs/models/*.yaml for VRAM and fertility.",
    )
    return parser.parse_args(argv)


def inspect_checkpoint(checkpoint_dir_str: str, output_path_str: str) -> None:
    ckpt_path = Path(checkpoint_dir_str).resolve()
    if not ckpt_path.is_dir():
        print(f"Notice: Checkpoint directory does not exist: {ckpt_path}. Skipping checkpoint spectral profile.")
        return

    print(f"\n=== Inspecting Checkpoint: {ckpt_path} ===")
    profile = inspect_checkpoint_spectral_profile(ckpt_path)

    print(f"Model Type:            {profile.model_type}")
    print(
        f"Layers:                {profile.num_hidden_layers} "
        f"| Hidden: {profile.hidden_size} "
        f"| Heads: {profile.num_attention_heads} (KV: {profile.num_key_value_heads}) "
        f"| Head Dim: {profile.head_dim}"
    )
    print(f"Vocab Size:            {profile.vocab_size} | Tied Embeddings: {profile.tie_word_embeddings}")
    print(f"Total Tensors:         {profile.total_tensors} | Total Parameters: {profile.total_parameters:,}")
    print(f"Protected Tensors:     {len(profile.protected_tensor_names)} | Editable Tensors: {len(profile.editable_tensor_names)}")
    print(f"Base Checkpoint SHA:   {profile.checkpoint_sha256[:16]}...")

    print("\n--- Module Type Spectral Averages ---")
    header = (
        f"{'Module Type':<35} {'Count':>6} {'Frob Norm':>10} {'L_inf':>10} "
        f"{'Cond Num':>10} {'Eff Rank':>10} {'R80%':>6}"
    )
    print(header)
    print("-" * len(header))
    for mod_type, stats in sorted(profile.module_type_averages.items()):
        cond_str = f"{stats['avg_condition_number']:.2f}" if "avg_condition_number" in stats else "N/A"
        erank_str = f"{stats['avg_effective_rank']:.2f}" if "avg_effective_rank" in stats else "N/A"
        r80_str = f"{stats['avg_energy_rank_80pct']:.1f}" if "avg_energy_rank_80pct" in stats else "N/A"
        print(
            f"{mod_type:<35} {stats['count']:>6} {stats['avg_frobenius']:>10.4f} {stats['avg_linf']:>10.4f} "
            f"{cond_str:>10} {erank_str:>10} {r80_str:>6}"
        )

    out_file = Path(output_path_str).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved checkpoint spectral profile to: {out_file}")


def compare_candidates(repo_root: Path) -> None:
    summary: dict[str, Any] = {
        "hardware": to_dict(get_hardware_status()),
        "candidates": {},
    }

    config_files = sorted((repo_root / "configs" / "models").glob("*.yaml"))
    if not config_files:
        print("No candidate config files found in configs/models/*.yaml.")
        return

    print("\n=== Comparing Candidate Models ===")
    for config_path in config_files:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_id = config["model_id"]
        print(f"\n--- Candidate: {model_id} ({config_path.name}) ---")
        try:
            memory = estimate_vram_from_config(
                model_id=model_id,
                approx_param_count=int(config["approx_param_count"]),
                dtype=config.get("native_dtype", "bfloat16"),
                seq_len=int(config.get("eval_max_model_length", 4096)),
            )
            fertility = compute_filipino_fertility(
                model_id=model_id,
                sentences=SAMPLE_FILIPINO_SENTENCES,
                probe_words=PROBE_WORDS,
            )
            report: dict[str, Any] = {
                "config_file": config_path.name,
                "license_name": config["license_name"],
                "memory_estimate": to_dict(memory),
                "fertility_report": to_dict(fertility),
            }
            summary["candidates"][model_id] = report
            print(json.dumps(report, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"Warning: Failed to inspect candidate {model_id}: {exc}")
            summary["candidates"][model_id] = {
                "config_file": config_path.name,
                "license_name": config.get("license_name", "unknown"),
                "error": str(exc),
            }

    output_path = repo_root / "experiments" / "candidate_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nSaved candidate comparison report to: {output_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    print("=== Local Hardware Status ===")
    hw_status = get_hardware_status()
    print(json.dumps(to_dict(hw_status), indent=2))

    if args.checkpoint_dir:
        inspect_checkpoint(args.checkpoint_dir, args.output)

    if args.compare_candidates:
        compare_candidates(repo_root)


if __name__ == "__main__":
    main()
