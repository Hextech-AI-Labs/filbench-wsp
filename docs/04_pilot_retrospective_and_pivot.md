# 04 — Pilot Retrospective: What Didn't Work, and the Pivot

Status: design-level retrospective, 2026-11-18.
Scope: findings from inspecting the operator algebra, prompt packets, and three
pilot proposals. **No FilBench score deltas between conditions are claimed yet**
(`experiments/` contains no committed results). The diagnosis below was made
*before* spending GPU-weeks on full evaluations — which is itself the point.

## 1. What we built (and keep)

A fail-closed harness for LLM-proposed weight edits:
- Declarative op schema with a hard safety budget (≤32 ops, ≤24 tensors,
  |δ| ≤ 0.15, per-tensor ΔF/F ≤ 0.10, ΔL∞/L∞ ≤ 0.25).
- fp32 delta computation with a post-bf16-rounding audit (no silent no-ops).
- Structure- and Frobenius-matched random controls (the "monkey" baseline).
- Tied-embedding protection, checkpoint hashing, tamper checks, cross-condition
  input parity, per-example paired statistics with sign tests and bootstrap CIs.

This harness is the project's main asset. Everything below is about the
*intervention space* we gave the proposer, not the harness.

## 2. The three blind pilots

| Pilot | Proposer | Target | Ops | Character of the plan |
| --- | --- | --- | --- | --- |
| pilot01 | claude-sonnet-5.5-high | Sailor2-1B | 3 | Whole-tensor o_proj gain +0.03125 at layers 15/23/31 |
| pilot02 | gpt-6.1-astra-max | Sailor2-1B | 8 | v_proj/o_proj gains +0.03–0.04 (layers 23–39), top-band down_proj boosts +0.04–0.05 (layers 40/44/45) |
| pilot01-extra (clean) | claude-sonnet-5.5-extra | Qwen3-1.7B-Base | 11 | Mid-band (64–1024) down_proj/up_proj boosts +0.03–0.04 (layers 14–24), v_proj bands, model.norm +0.02 |

Note: the two `rejection_report.json` files in `configs/plans/` are
`FileExistsError` re-run collisions, not scientific rejections. The plans
themselves passed validation.

All three proposers independently converged on the same generic recipe:
"small positive gains on mid-to-late layers." Their own hypotheses admit it is
"a zero-shot guess with no gradient or eval feedback."

## 3. What didn't work (found by inspection, before evaluation)

**F1 — The intervention space is language-blind.**
Every allowed op (tensor/row/column/spectral-band scaling) and every statistic
in the prompt packet (norms, effective rank, condition numbers) is identical
whether the target capability is Filipino, French, or chess. Nothing in the
packet tells the proposer where Filipino actually lives in the weights. The
matched random control has the same structure and norm, so the experiment can
only detect "did the LLM guess layers/bands better than chance" — a weak bet.

**F2 — One proposed op is a mathematical no-op.**
The accepted Qwen3 plan scales `model.norm.weight` by +0.02. A uniform
final-norm gain multiplies all logits equally: a temperature change. Argmax is
invariant, and our frozen eval uses greedy decoding (temperature 0). It cannot
change any answer except via bf16 rounding ties. A wasted op that *looked*
mechanistically plausible in the rationale.

**F3 — Generic gains amplify all languages equally.**
Whole-tensor v_proj/o_proj scaling raises a layer's contribution to the
residual stream for English and Filipino alike. There is no mechanism by which
it should preferentially help Filipino.

**F4 — The measurement floor is noisy where we hoped to see gains.**
Several dev tasks have tiny n (e.g., scores of 5.00% = 1/20 correct); one
flipped answer moves a task by 5 points. Generation scores are near floor
(tatoeba_tgl 16.7, ntrex128 15.4, tico19 12.9 rougeL) largely because a base
model is never *shown* the translation task in zero-shot format — a prompting
limitation, not a weights limitation. Weight edits cannot teach task format.

**F5 — One-shot, no feedback.**
Each proposer gets one packet and one answer. It never learns whether it was
warm or cold, so proposals cannot improve within a session.

## 4. The prediction this leads to

Blind, language-agnostic gain edits should land within ~0–2 points of baseline
— statistically indistinguishable from the matched random control. Running the
full three-condition eval on these plans would most likely buy an expensive
null result. So we pivoted the *inputs*, not the harness.

## 5. The pivot

1. **Measure before editing** (`scripts/06_language_diagnostics.py`):
   forward Filipino vs English text through the base model, rank neurons and
   layers by language selectivity, and ship those measurements in packet v2.
   The proposer now aims at Filipino-enriched subspaces instead of guessing.
   Still no gradients, no training — the "training-free" claim is preserved.
2. **Counterfactual controls** (`scripts/07_counterfactual_controls.py`):
   sign-flip (same knobs, opposite direction) and layer-shuffle (same edits,
   random layers). If the original helps, the flip hurts, and the shuffle does
   nothing, the LLM's *choices* carry signal. Cheap, and far more convincing
   than beating the monkey alone.
3. **Throughput**: configurable eval batch size (re-run baseline for parity).
4. **Next**: a tied-safe Filipino token-row op (language-prior lever), and an
   optional feedback condition (K plans → cheap screen → revise), kept separate
   from the headline zero-shot condition.

## 6. Talk narrative (five beats)

1. **The bet**: can a frontier LLM rewire a small model's weights — no training
   — to make it better at Filipino?
2. **We built the referee first**: budgets, audits, hashing, matched random
   controls, paired stats. If the LLM cheats or lucks out, we will know.
3. **Three frontier models, one blind guess each**: all proposed the same
   generic "turn mid-late layers up a little." One op provably does nothing.
4. **The diagnosis**: our operator alphabet had no way to say "Filipino."
   The packet described the model's *math*, not its *languages*. The monkey
   was always going to tie.
5. **The pivot**: show the model Filipino text, measure which neurons light
   up, hand the proposer a map instead of a spec sheet — and add upside-down
   and shuffled controls so any win is attributable to the LLM's choices.

One-line takeaway: *the hard part of weight-space programming isn't applying
edits safely — it's giving the editor a reason to aim.*

## 7. What counts as a win

Dev: pooled paired delta with bootstrap CI excluding zero, beating random,
sign-flip, and layer-shuffle controls, with no fluency collapse. Only then
unseal the test split, once. A clean null on blind editing remains a valid,
reportable result.