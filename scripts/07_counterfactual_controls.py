"""Derive counterfactual control plans (sign-flip, layer-shuffle) from an accepted
edit bundle and materialize them through the existing fail-closed applier.

If the LLM's choices carry signal: sign-flip should hurt, layer-shuffle should
do nothing, while the original plan helps. All controls reuse the same safety
budget, fp32/bf16 audit, and checkpoint hashing as the main pipeline.

Usage (repo root, single line):
  python scripts/07_counterfactual_controls.py --bundle configs/plans/pilot01_sonnet55extra_qwen3_1_7b_clean/edit_bundle.json --base-checkpoint checkpoints/base/qwen3_1_7b_base --seed 20261118
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from filbench_wsp.editing import EditPlan, apply_edit_plan_to_checkpoint
from filbench_wsp.manifest import compute_checkpoint_hash

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def derive_signflip(plan_dict: dict, new_id: str) -> dict:
    d = copy.deepcopy(plan_dict)
    d["plan_id"] = new_id
    d["hypothesis"] = f"[COUNTERFACTUAL sign-flip of {plan_dict['plan_id']}] " + d.get("hypothesis", "")
    for op in d["ops"]:
        op["delta_scale"] = -float(op["delta_scale"])
        op["rationale"] = "[sign-flip control] " + op.get("rationale", "")
    d.setdefault("proposer_metadata", {})["control_type"] = "sign_flip"
    d["proposer_metadata"]["derived_from_plan"] = plan_dict["plan_id"]
    return d


def derive_layershuffle(plan_dict: dict, new_id: str, num_layers: int, seed: int) -> tuple[dict, dict]:
    d = copy.deepcopy(plan_dict)
    src = sorted({int(m.group(1)) for op in d["ops"] if (m := LAYER_RE.search(op["tensor_name"]))})
    if not src:
        raise ValueError("Plan targets no model.layers.* tensors; layer-shuffle is not meaningful.")
    if len(src) > num_layers:
        raise ValueError("More source layers than model layers; cannot shuffle.")
    rng = random.Random(seed)
    mapping = dict(zip(src, rng.sample(range(num_layers), len(src))))
    for op in d["ops"]:
        m = LAYER_RE.search(op["tensor_name"])
        if m:
            i = int(m.group(1))
            op["tensor_name"] = LAYER_RE.sub(f"model.layers.{mapping[i]}.", op["tensor_name"], count=1)
            op["rationale"] = f"[layer-shuffle control {i}->{mapping[i]}] " + op.get("rationale", "")
    d["plan_id"] = new_id
    d["hypothesis"] = (f"[COUNTERFACTUAL layer-shuffle of {plan_dict['plan_id']}; mapping {mapping}] "
                       + d.get("hypothesis", ""))
    d.setdefault("proposer_metadata", {})["control_type"] = "layer_shuffle"
    d["proposer_metadata"]["derived_from_plan"] = plan_dict["plan_id"]
    d["proposer_metadata"]["layer_mapping"] = mapping
    d["proposer_metadata"]["shuffle_seed"] = seed
    return d, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True, help="Accepted edit_bundle.json of the LLM plan")
    ap.add_argument("--base-checkpoint", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--out-root", type=Path, default=Path("checkpoints/edited"))
    ap.add_argument("--bundle-root", type=Path, default=Path("configs/plans"))
    ap.add_argument("--with-random", action="store_true", help="Also materialize a matched random control per counterfactual")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("status") != "accepted":
        raise ValueError("Bundle is not an accepted plan.")
    plan_dict = bundle["plan"]
    orig_id = plan_dict["plan_id"]

    expected_sha = bundle["llm_edited_report"]["base_checkpoint_sha256"]
    actual_sha = compute_checkpoint_hash(args.base_checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Base checkpoint hash mismatch: bundle expects {expected_sha[:16]}..., "
            f"got {actual_sha[:16]}.... Point --base-checkpoint at the exact same base weights."
        )

    cfg = json.loads((args.base_checkpoint / "config.json").read_text(encoding="utf-8"))
    num_layers = int(cfg["num_hidden_layers"])

    controls = [("signflip", derive_signflip(plan_dict, f"{orig_id}_signflip"))]
    shuffled, mapping = derive_layershuffle(plan_dict, f"{orig_id}_layershuffle", num_layers, args.seed)
    controls.append(("layershuffle", shuffled))
    print(f"Layer-shuffle mapping: {mapping}")

    for label, d in controls:
        plan = EditPlan.from_dict(d)  # re-validates the full fail-closed schema/budget
        out_dir = args.out_root / plan.plan_id
        rep = apply_edit_plan_to_checkpoint(
            plan=plan, base_checkpoint_dir=args.base_checkpoint,
            output_checkpoint_dir=out_dir, mode="llm_edited", force=args.force,
        )
        bundle_dir = args.bundle_root / plan.plan_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        record = {"status": "accepted", "control_type": label, "derived_from_plan": orig_id,
                  "seed": args.seed, "plan": plan.to_dict(), "execution_report": rep.to_dict()}
        (bundle_dir / "edit_bundle.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "filbench_wsp_edit_report.json").write_text(
            json.dumps({"plan": plan.to_dict(), "execution_report": rep.to_dict()}, indent=2),
            encoding="utf-8")
        print(f"[OK] {label}: {out_dir}")
        print(f"     audit: {bundle_dir / 'edit_bundle.json'}")
        if args.with_random:
            rdir = args.out_root / f"{plan.plan_id}_random_control"
            apply_edit_plan_to_checkpoint(
                plan=plan, base_checkpoint_dir=args.base_checkpoint,
                output_checkpoint_dir=rdir, mode="random_control",
                random_seed=args.seed + 1, force=args.force,
            )
            print(f"[OK] matched random: {rdir}")
        print(f"     eval: python scripts/03_run_eval.py --exp-id {plan.plan_id}_dev "
              f"--condition llm_edited --base-checkpoint-dir {args.base_checkpoint} "
              f"--eval-checkpoint-dir {out_dir} --base-model-name {plan.target_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())