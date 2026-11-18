"""In-memory multi-plan screening (no checkpoint materialization).

Loads the base model once, applies each candidate plan's validated deltas in
RAM, measures per-token NLL on parallel Filipino/English sentences, restores
the base weights, and moves to the next plan. Ranks plans by Filipino NLL
improvement subject to an English-regression guard, and can render a feedback
packet for a second proposer round.

Accepts raw chatbot responses (.txt), plan JSONs, or accepted edit bundles.

Usage (repo root, single line):
  python scripts/08_screen_plans.py --base-checkpoint checkpoints/base/sailor2_1b --plans prompts/responses/plan_a.txt prompts/responses/plan_b.txt --allow-tied-rows --packet-in prompts/packet_v2_dev_zero_shot.md --feedback-out prompts/packet_v2_feedback_round2.md --report-out experiments/screen_round1.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from filbench_wsp.editing.applier import (
    compute_and_validate_tensor_edit,
    compute_and_validate_tied_pair_edit,
)
from filbench_wsp.editing.schema import EditPlan, EditValidationError
from filbench_wsp.proposer import extract_json_from_raw_response


def _load_seed_corpora() -> tuple[list[str], list[str]]:
    path = Path(__file__).with_name("06_language_diagnostics.py")
    spec = importlib.util.spec_from_file_location("filbench_wsp_lang_diag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FIL_SEED_SENTENCES, mod.EN_SEED_SENTENCES


def load_plan(path: Path, allow_tied_rows: bool) -> EditPlan:
    if path.suffix.lower() == ".txt":
        d = extract_json_from_raw_response(path.read_text(encoding="utf-8"))
    else:
        d = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(d, dict) and d.get("status") == "accepted" and "plan" in d:
            d = d["plan"]
    if not d.get("plan_id"):
        d["plan_id"] = path.stem
    if allow_tied_rows:
        b = dict(d.get("budget") or {})
        b["allow_tied_embedding_rows"] = True
        d["budget"] = b
    return EditPlan.from_dict(d)


def mean_nll(model, tok, sentences, device, max_len: int = 256) -> float:
    import torch

    total, n = 0.0, 0
    with torch.no_grad():
        for s in sentences:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            if ids.input_ids.shape[1] < 2:
                continue
            out = model(**ids, use_cache=False)
            logits = out.logits[:, :-1, :].float()
            tgt = ids.input_ids[:, 1:]
            total += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"
            ).item()
            n += int(tgt.numel())
    return total / max(n, 1)


def apply_plan_in_memory(model, plan: EditPlan) -> dict:
    """Return {key: original_tensor} stash after loading validated edits; caller restores."""
    import torch

    sd = model.state_dict()
    ops_by_tensor = defaultdict(list)
    for op in plan.ops:
        ops_by_tensor[op.tensor_name].append(op)
    tied_mode = any(op.op_type == "scale_tied_rows" for op in plan.ops)
    tie = bool(getattr(model.config, "tie_word_embeddings", False))

    edited: dict[str, torch.Tensor] = {}
    for name, ops in ops_by_tensor.items():
        if name not in sd:
            raise EditValidationError(f"Tensor {name!r} not in model state dict")
        if tied_mode and name == "model.embed_tokens.weight":
            lm_base = sd.get("lm_head.weight") if tie else None
            e, _, l, _ = compute_and_validate_tied_pair_edit(
                embed_base=sd[name], lm_base=lm_base, ops=ops,
                budget=plan.budget, mode="llm_edited",
            )
            edited[name] = e  # tied param object: loading embed updates lm_head too
        else:
            e, _ = compute_and_validate_tensor_edit(
                tensor_name=name, base_weight=sd[name], ops=ops,
                budget=plan.budget, mode="llm_edited",
            )
            edited[name] = e

    stash = {k: sd[k].detach().clone() for k in edited}
    model.load_state_dict(edited, strict=False)
    if tied_mode and tie and "lm_head.weight" in sd:
        if not torch.equal(model.state_dict()["lm_head.weight"], edited["model.embed_tokens.weight"]):
            raise RuntimeError("Tied lm_head did not follow embedding edit in memory")
    return stash


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--plans", nargs="+", type=Path, required=True)
    ap.add_argument("--allow-tied-rows", action="store_true")
    ap.add_argument("--fil-file", default=None)
    ap.add_argument("--en-file", default=None)
    ap.add_argument("--max-en-regression", type=float, default=0.05,
                    help="Max allowed English NLL increase (nats/token) for a plan to pass the guard")
    ap.add_argument("--report-out", type=Path, default=Path("experiments/screen_report.json"))
    ap.add_argument("--packet-in", type=Path, default=None)
    ap.add_argument("--feedback-out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fil_seed, en_seed = _load_seed_corpora()
    fil = ([l.strip() for l in Path(args.fil_file).read_text(encoding="utf-8").splitlines() if l.strip()]
           if args.fil_file else fil_seed)
    en = ([l.strip() for l in Path(args.en_file).read_text(encoding="utf-8").splitlines() if l.strip()]
          if args.en_file else en_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint, torch_dtype=dtype).to(device).eval()

    base_fil = mean_nll(model, tok, fil, device)
    base_en = mean_nll(model, tok, en, device)
    print(f"BASE  fil_nll={base_fil:.4f}  en_nll={base_en:.4f}  margin(fil-en)={base_fil - base_en:+.4f}")

    results = []
    for path in args.plans:
        entry = {"plan_source": str(path), "status": "ok"}
        stash = None
        try:
            plan = load_plan(path, args.allow_tied_rows)
            entry["plan_id"] = plan.plan_id
            entry["num_ops"] = len(plan.ops)
            stash = apply_plan_in_memory(model, plan)
            f = mean_nll(model, tok, fil, device)
            e = mean_nll(model, tok, en, device)
            entry.update(
                fil_nll=round(f, 6), en_nll=round(e, 6),
                fil_delta=round(f - base_fil, 6), en_delta=round(e - base_en, 6),
                margin_delta=round((f - e) - (base_fil - base_en), 6),
            )
            entry["passes_en_guard"] = entry["en_delta"] <= args.max_en_regression
            entry["improves_fil"] = entry["fil_delta"] < 0.0
            print(f"[{plan.plan_id}] fil_delta={entry['fil_delta']:+.4f} "
                  f"en_delta={entry['en_delta']:+.4f} guard={'PASS' if entry['passes_en_guard'] else 'FAIL'}")
        except Exception as exc:
            entry.update(status="rejected", error_type=type(exc).__name__, error_message=str(exc))
            print(f"[{path.name}] REJECTED: {exc}")
        finally:
            if stash is not None:
                model.load_state_dict(stash, strict=False)
        results.append(entry)

    ok = [r for r in results if r["status"] == "ok" and r["passes_en_guard"] and r["improves_fil"]]
    ok.sort(key=lambda r: r["fil_delta"])
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": args.base_checkpoint,
        "base_fil_nll": round(base_fil, 6), "base_en_nll": round(base_en, 6),
        "max_en_regression": args.max_en_regression,
        "n_fil_sentences": len(fil), "n_en_sentences": len(en),
        "results": results,
        "ranking": [r["plan_id"] for r in ok],
        "note": "Proxy screen only (parallel-corpus NLL). Not a FilBench score. "
                "Materialize the winner via scripts/05 and run the full dev eval.",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved screen report: {args.report_out}")
    print("Ranking:", report["ranking"] or "(no plan passed both guards)")

    if args.packet_in and args.feedback_out:
        text = args.packet_in.read_text(encoding="utf-8")
        meta = json.loads(args.packet_in.with_suffix(".meta.json").read_text(encoding="utf-8"))
        lines = ["", "## 7. Screening Feedback (previous round) — revise your plan",
                 "Each candidate below was applied in-memory to the base model and scored by "
                 "per-token NLL on parallel Filipino/English sentences (lower = better). "
                 "fil_delta<0 means Filipino text became more predictable; en_delta>0 means "
                 "English fluency regressed. This is a proxy, not FilBench.",
                 "", "| Plan | fil_delta | en_delta | EN guard |", "| :--- | :---: | :---: | :---: |"]
        for r in results:
            if r["status"] == "ok":
                lines.append(f"| `{r['plan_id']}` | {r['fil_delta']:+.4f} | {r['en_delta']:+.4f} "
                             f"| {'PASS' if r['passes_en_guard'] else 'FAIL'} |")
            else:
                lines.append(f"| `{r.get('plan_id', Path(r['plan_source']).stem)}` | rejected | — | {r['error_type']} |")
        lines += ["", "Propose ONE revised plan (same JSON schema). Keep what helped, remove what "
                  "hurt or was rejected, and explain the revision in `hypothesis`."]
        new_text = text + "\n".join(lines) + "\n"
        new_id = meta.get("packet_id", "packet") + "_feedback"
        import re
        new_text = re.sub(r"Packet ID: `[^`]+`", f"Packet ID: `{new_id}`", new_text, count=1)
        args.feedback_out.parent.mkdir(parents=True, exist_ok=True)
        args.feedback_out.write_text(new_text, encoding="utf-8")
        meta.update(packet_id=new_id,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    prompt_markdown_path=str(args.feedback_out),
                    prompt_sha256=hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
                    screen_report_path=str(args.report_out))
        args.feedback_out.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Rendered feedback packet: {args.feedback_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())