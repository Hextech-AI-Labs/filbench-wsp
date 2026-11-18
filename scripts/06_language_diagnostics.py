"""Language-contrastive activation diagnostics ("brain scan") + packet v2 renderer.

Runs the BASE model on Filipino vs English text and records which MLP neurons
respond more to Filipino. No gradients, no training, no weight changes.
Outputs:
  1. configs/<name>_language_profile.json  (machine-readable profile)
  2. prompts/packet_v2_*.md + .meta.json   (upgraded packet, ready for the chatbot)

Usage (repo root, single line):
  python scripts/06_language_diagnostics.py --checkpoint checkpoints/base/sailor2_1b --packet-in prompts/packet_v1_dev_zero_shot.md --packet-out prompts/packet_v2_dev_zero_shot.md --output-json configs/sailor2_1b_language_profile.json --allow-tied-rows
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIL_SEED_SENTENCES = [
    "Ang Pilipinas ay isang arkipelago na binubuo ng mahigit pitong libong isla.",
    "Bumaha sa maraming lugar matapos ang walang tigil na pag-ulan kahapon.",
    "Ayon sa Department of Health, libre ang bakuna para sa lahat ng mga sanggol.",
    "Mahilig kumain ng adobo at sinigang ang karamihan sa mga pamilyang Pilipino.",
    "Ipinagdiriwang ang Araw ng Kalayaan tuwing ika-dose ng Hunyo.",
    "Nag-aaral ang mga bata sa paaralan mula Lunes hanggang Biyernes.",
    "Bumili si Nanay ng isda at gulay sa palengke kaninang umaga.",
    "Mainit ang panahon sa Mayo kaya maraming tao ang pumupunta sa dagat.",
    "Ang wikang Filipino ay isa sa mga opisyal na wika ng bansa.",
    "Nanalo ang pambansang koponan sa basketball kagabi.",
    "Kailangan nating magtipid ng tubig lalo na sa tag-init.",
    "Nagtrabaho ang kanyang ama sa ibang bansa upang matustusan ang pamilya.",
    "Maraming turista ang bumibisita sa Boracay dahil sa puting buhangin.",
    "Binasa ng guro ang kwento tungkol sa mga bayani ng rebolusyon.",
    "Masarap ang halo-halo lalo na kapag mainit ang hapon.",
    "Nagdaos ng pista ang barangay bilang pasasalamat sa masaganang ani.",
    "Dapat tayong maghugas ng kamay bago kumain upang maiwasan ang sakit.",
    "Umakyat ang presyo ng bigas sa merkado nitong nakaraang buwan.",
    "Ang mga magsasaka ay umaasa sa ulan para sa kanilang mga pananim.",
    "Ipinanganak siya sa isang maliit na bayan sa lalawigan ng Quezon.",
    "Tinulungan ng mga boluntaryo ang mga pamilyang naapektuhan ng bagyo.",
    "Mahalaga ang edukasyon para sa kinabukasan ng bawat kabataan.",
    "Nagsimula ang klase alas-siyete ng umaga at natapos alas-singko ng hapon.",
    "Ang jeepney ay isa sa mga pinakasikat na sasakyan sa lungsod.",
    "Nagtipon ang mga residente upang pag-usapan ang problema sa basura.",
    "Masaya ang buong pamilya noong muling nakauwi ang kanilang anak.",
    "Itinanim nila ang mga puno sa gilid ng ilog upang maiwasan ang pagguho ng lupa.",
    "Binigyang-diin ng alkalde ang kahalagahan ng kaligtasan sa panahon ng bagyo.",
]

EN_SEED_SENTENCES = [
    "The Philippines is an archipelago made up of more than seven thousand islands.",
    "Many areas were flooded after the nonstop rain yesterday.",
    "According to the Department of Health, vaccines are free for all infants.",
    "Most families enjoy eating roasted chicken and sour soup on weekends.",
    "Independence Day is celebrated every twelfth of June.",
    "Children study at school from Monday to Friday.",
    "Mother bought fish and vegetables at the market this morning.",
    "The weather is hot in May, so many people go to the beach.",
    "English is one of the official languages of the country.",
    "The national team won the basketball game last night.",
    "We need to save water, especially during the dry season.",
    "His father worked abroad to support the family.",
    "Many tourists visit the island because of its white sand.",
    "The teacher read a story about the heroes of the revolution.",
    "The dessert is delicious, especially on a warm afternoon.",
    "The village held a festival to give thanks for the abundant harvest.",
    "We should wash our hands before eating to avoid sickness.",
    "The price of rice in the market rose last month.",
    "Farmers depend on the rain for their crops.",
    "She was born in a small town in the province.",
    "Volunteers helped the families affected by the typhoon.",
    "Education is important for the future of every young person.",
    "The class started at seven in the morning and ended at five in the afternoon.",
    "The bus is one of the most popular vehicles in the city.",
    "Residents gathered to discuss the garbage problem.",
    "The whole family was happy when their child came home again.",
    "They planted trees along the river to prevent landslides.",
    "The mayor emphasized the importance of safety during storms.",
]

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def load_sentences(path: str | None, seed: list[str]) -> list[str]:
    if path:
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"No sentences found in {path}")
        return lines
    return list(seed)


def best_windows(scores: list[float], width: int, top: int = 3) -> list[dict]:
    n = len(scores)
    if width > n:
        return []
    pref = [0.0]
    for s in scores:
        pref.append(pref[-1] + s)
    wins = [((pref[i + width] - pref[i]) / width, i, i + width) for i in range(n - width + 1)]
    wins.sort(reverse=True)
    return [{"start": i, "end": j, "mean_selectivity": round(m, 4)} for m, i, j in wins[:top]]


def collect_profile(checkpoint: str, fil: list[str], en: list[str], max_len: int = 256) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()

    names = [n for n, m in model.named_modules() if n.endswith("mlp.down_proj")]
    if not names:
        raise RuntimeError("No mlp.down_proj modules found; unsupported architecture.")
    stats = {n: {"fil": [None, None, 0], "en": [None, None, 0]} for n in names}
    modules = dict(model.named_modules())

    def make_hook(name: str, lang: str):
        def hook(mod, inp, out):
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = stats[name][lang]
            sums, sqs = a.sum(0), (a * a).sum(0)
            s[0] = sums if s[0] is None else s[0] + sums
            s[1] = sqs if s[1] is None else s[1] + sqs
            s[2] += a.shape[0]
        return hook

    with torch.no_grad():
        for lang, sentences in (("fil", fil), ("en", en)):
            handles = [modules[n].register_forward_hook(make_hook(n, lang)) for n in names]
            for s in sentences:
                ids = tok(s, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                model(**ids, use_cache=False)
            for h in handles:
                h.remove()

    layers, all_neurons = [], []
    for name in names:
        m = LAYER_RE.search(name)
        layer_idx = int(m.group(1)) if m else len(layers)
        sf, qf, nf = stats[name]["fil"]
        se, qe, ne = stats[name]["en"]
        mu_f, mu_e = sf / nf, se / ne
        var_f = (qf / nf - mu_f * mu_f).clamp_min(0)
        var_e = (qe / ne - mu_e * mu_e).clamp_min(0)
        d = ((mu_f - mu_e) / torch.sqrt(0.5 * (var_f + var_e) + 1e-8)).tolist()
        mu_f_l, mu_e_l = mu_f.tolist(), mu_e.tolist()
        order = sorted(range(len(d)), key=lambda i: d[i], reverse=True)
        top8_fil = sum(d[i] for i in order[:8]) / 8
        top8_en = sum(-d[i] for i in order[-8:]) / 8
        layers.append({
            "layer": layer_idx, "module": name,
            "top8_fil_selectivity": round(top8_fil, 4),
            "top8_en_selectivity": round(top8_en, 4),
            "n_neurons_d_gt_0.5": sum(1 for x in d if x > 0.5),
            "windows_256": best_windows(d, 256),
            "windows_64": best_windows(d, 64),
        })
        for i in order[:16]:
            all_neurons.append({"layer": layer_idx, "neuron": i,
                                "selectivity": round(d[i], 4),
                                "mean_fil": round(mu_f_l[i], 4),
                                "mean_en": round(mu_e_l[i], 4)})
    layers.sort(key=lambda x: x["layer"])
    all_neurons.sort(key=lambda x: x["selectivity"], reverse=True)

    cf, ce = Counter(), Counter()
    for s in fil:
        cf.update(tok(s, add_special_tokens=False).input_ids)
    for s in en:
        ce.update(tok(s, add_special_tokens=False).input_ids)
    cand = [(t, c) for t, c in cf.items() if c >= 5]
    cand.sort(key=lambda tc: ((tc[1] + 1) / (ce[tc[0]] + 1), tc[1]), reverse=True)
    tokens = [{"token_id": t, "text": tok.decode([t]), "count_fil": c, "count_en": ce[t]}
              for t, c in cand[:64]]

    return {
        "checkpoint": checkpoint,
        "model_type": getattr(model.config, "model_type", "unknown"),
        "n_fil_sentences": len(fil), "n_en_sentences": len(en),
        "n_fil_tokens": stats[names[0]]["fil"][2], "n_en_tokens": stats[names[0]]["en"][2],
        "layers": layers,
        "top_neurons": all_neurons[:48],
        "filipino_distinctive_tokens": tokens,
        "method": "Per-neuron mean activation at mlp.down_proj input; d=(mu_fil-mu_en)/sqrt((var_fil+var_en)/2+1e-8).",
    }


def render_section(p: dict, baseline_counts: dict | None, allow_tied_rows: bool = False) -> str:
    L = []
    L.append("## 3b. Language-Contrastive Activation Profile (measured, not guessed)")
    L.append(f"Measured on the untouched base checkpoint by forwarding {p['n_fil_sentences']} Filipino "
             f"({p['n_fil_tokens']:,} tokens) and {p['n_en_sentences']} English ({p['n_en_tokens']:,} tokens) "
             "sentences. Selectivity d>0 means the neuron responds more to Filipino. "
             "Pilot corpus is small: treat rankings as strong hints, not ground truth.")
    L.append("")
    L.append("### Layer ranking (MLP Filipino selectivity)")
    L.append("| Layer | Top-8 Fil d | Top-8 En d | #neurons d>0.5 | Best 256-wide Fil window |")
    L.append("| :---: | :---: | :---: | :---: | :--- |")
    for row in sorted(p["layers"], key=lambda r: r["top8_fil_selectivity"], reverse=True):
        w = row["windows_256"][0] if row["windows_256"] else None
        wtxt = f"[{w['start']}, {w['end']}) mean d={w['mean_selectivity']}" if w else "n/a"
        L.append(f"| {row['layer']} | {row['top8_fil_selectivity']} | {row['top8_en_selectivity']} "
                 f"| {row['n_neurons_d_gt_0.5']} | {wtxt} |")
    L.append("")
    L.append("### Top Filipino-selective neurons")
    L.append("Neuron i of layer L = column i of `model.layers.L.mlp.down_proj.weight` "
             "= row i of `model.layers.L.mlp.up_proj.weight` / `gate_proj.weight`.")
    L.append("| Layer | Neuron | d | mean Fil | mean En |")
    L.append("| :---: | :---: | :---: | :---: | :---: |")
    for t in p["top_neurons"][:24]:
        L.append(f"| {t['layer']} | {t['neuron']} | {t['selectivity']} | {t['mean_fil']} | {t['mean_en']} |")
    L.append("")
    L.append("### How to use this profile within the op budget")
    L.append("- Prefer few grounded ops over many tiny ones. A single-neuron edit is functionally tiny; "
             "use the 256-wide windows above with `scale_cols` on `down_proj` (delta +0.05..+0.15).")
    L.append("- `down_proj` column scaling linearly amplifies that neuron's write into the residual stream. "
             "Optionally pair with `scale_rows` on `up_proj` for the same window. `gate_proj` is nonlinear (SiLU); optional.")
    L.append("- Consider `scale_tensor` +0.03..+0.08 on `down_proj` of the 2-3 highest-Filipino layers.")
    L.append("- DO NOT spend ops on `model.norm.weight`: a uniform final-norm scale is a temperature change; "
             "it cannot change argmax accuracy or greedy decoding (only bf16 rounding ties).")
    if allow_tied_rows:
        L.append("- The tied embedding table IS editable in this packet via the new `scale_tied_rows` "
                 "operator (section 3d below). Use it to raise the Filipino language prior.")
    else:
        L.append("- Embeddings/lm_head remain tied and protected. Token list below is informational only "
                 "(for a future tied-safe row op); do not propose edits to them.")
    L.append("")
    header = ("### Filipino-distinctive tokens (editable via scale_tied_rows)" if allow_tied_rows
              else "### Filipino-distinctive tokens (informational, protected)")
    L.append(header)
    L.append("| Token ID | Text | count Fil | count En |")
    L.append("| :---: | :--- | :---: | :---: |")
    for t in p["filipino_distinctive_tokens"][:32]:
        txt = t["text"].replace("|", "\\|").replace("\n", " ")
        L.append(f"| {t['token_id']} | `{txt}` | {t['count_fil']} | {t['count_en']} |")

    if baseline_counts:
        L.append("")
        L.append("## 3c. Dev sample sizes (noise awareness)")
        L.append("Small-n tasks are noisy: one flipped answer can move a task by several points. "
                 "Prioritize hypotheses that should move many tasks, and judge results on pooled/paired stats.")
        L.append("| Task | dev n |")
        L.append("| :--- | :---: |")
        for k, v in sorted(baseline_counts.items(), key=lambda kv: kv[1]):
            L.append(f"| `{k}` | {v} |")

    if allow_tied_rows:
        L.append("")
        L.append("## 3d. Additional Operator Enabled For This Packet: `scale_tied_rows`")
        L.append("This model ties input and output embeddings. `scale_tied_rows` applies ONE shared "
                 "row perturbation to both copies, preserving the tie (verified bitwise by the applier). "
                 "Mechanism: boosting Filipino-token rows raises the likelihood of Filipino continuations "
                 "relative to English (a language-prior shift), which can help MCF answer ranking and translation.")
        L.append("Rules:")
        L.append("- `tensor_name` MUST be exactly `model.embed_tokens.weight` (lm_head is mirrored automatically; never target it).")
        L.append("- `row_indices`: list of unique token IDs (use the table above; you may add other clearly Filipino tokens). Max 512 rows.")
        L.append("- `delta_scale`: positive, +0.02..+0.08 recommended. Do NOT set row/col/rank bounds.")
        L.append("- Do NOT include a `budget` object in your JSON; the harness injects the enabled budget.")
        L.append("- Keep total ops within the section-4 budget; combine with at most a few grounded MLP ops.")
        L.append("Example op:")
        L.append("```json")
        L.append('{"op_type": "scale_tied_rows", "tensor_name": "model.embed_tokens.weight", '
                 '"delta_scale": 0.05, "row_indices": [123, 456, 789], '
                 '"rationale": "Boost high-frequency Filipino function-word and content-word tokens to shift the language prior."}')
        L.append("```")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="BASE checkpoint dir or HF id (never an edited one)")
    ap.add_argument("--fil-file", default=None, help="Optional Filipino corpus, one sentence per line")
    ap.add_argument("--en-file", default=None, help="Optional English corpus, one sentence per line")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--packet-in", type=Path, default=None, help="Existing packet_v1 .md to upgrade")
    ap.add_argument("--packet-out", type=Path, default=None, help="Output packet_v2 .md")
    ap.add_argument("--allow-tied-rows", action="store_true",
                    help="Offer the tie-safe scale_tied_rows operator in the rendered packet")
    args = ap.parse_args()

    fil = load_sentences(args.fil_file, FIL_SEED_SENTENCES)
    en = load_sentences(args.en_file, EN_SEED_SENTENCES)
    profile = collect_profile(args.checkpoint, fil, en, args.max_len)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved language profile: {args.output_json}")
    top = sorted(profile["layers"], key=lambda r: r["top8_fil_selectivity"], reverse=True)[:5]
    print("Most Filipino-selective layers:", [(r["layer"], r["top8_fil_selectivity"]) for r in top])

    baseline_counts = None
    if args.packet_in:
        if not args.packet_out:
            raise ValueError("--packet-out is required when --packet-in is given")
        v1_text = args.packet_in.read_text(encoding="utf-8")
        meta_in_path = args.packet_in.with_suffix(".meta.json")
        meta = json.loads(meta_in_path.read_text(encoding="utf-8"))
        bs_path = Path(meta.get("baseline_summary_path", ""))
        if bs_path.exists():
            bs = json.loads(bs_path.read_text(encoding="utf-8"))
            baseline_counts = {k: v["effective"] for k, v in (bs.get("doc_counts_by_task") or {}).items()
                               if isinstance(v, dict) and "effective" in v}
        section = render_section(profile, baseline_counts, allow_tied_rows=args.allow_tied_rows)
        new_id = meta.get("packet_id", "packet_v1") + "_langv2"
        v2_text = re.sub(r"Packet ID: `[^`]+`", f"Packet ID: `{new_id}`", v1_text, count=1)
        idx = v2_text.find("\n## 4.")
        if idx == -1:
            v2_text = v2_text + "\n" + section
        else:
            v2_text = v2_text[:idx] + "\n" + section + v2_text[idx:]
        args.packet_out.parent.mkdir(parents=True, exist_ok=True)
        args.packet_out.write_text(v2_text, encoding="utf-8")
        meta.update(
            packet_id=new_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            prompt_markdown_path=str(args.packet_out),
            prompt_sha256=hashlib.sha256(v2_text.encode("utf-8")).hexdigest(),
            language_profile_path=str(args.output_json),
            language_profile_sha256=hashlib.sha256(args.output_json.read_bytes()).hexdigest(),
            allow_tied_rows=bool(args.allow_tied_rows),
        )
        args.packet_out.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Rendered packet v2: {args.packet_out}")
        print(f"  meta: {args.packet_out.with_suffix('.meta.json')}")
    else:
        section_path = args.output_json.with_suffix(".section.md")
        section_path.write_text(
            render_section(profile, None, allow_tied_rows=args.allow_tied_rows), encoding="utf-8")
        print(f"Saved standalone packet section: {section_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())