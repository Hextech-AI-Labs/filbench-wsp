# Direct Weight-Space Intervention Task (FilBench-WSP)
Packet ID: `packet_v1_qwen3_1_7b_dev_zero_shot`

You are participating in a scientific experiment testing whether a frontier LLM can propose a **direct, training-free weight perturbation** ($\theta_{\text{edited}} = \theta_{\text{base}} + \Delta\theta$) to improve the Filipino/Tagalog/Cebuano performance of `Qwen/Qwen3-1.7B-Base` on the FilBench development split, without gradient descent, fine-tuning, or adapters.

---

## 1. Target Model Architecture (`Qwen/Qwen3-1.7B-Base`)
- **Architecture:** `Qwen3ForCausalLM` (`bfloat16`)
- **Depth & Width:** `28` transformer layers (`model.layers.0` .. `model.layers.27`), `hidden_size = 2048`
- **Attention:** `16` query heads, `8` key/value heads, `head_dim = 128`
- **Per-Layer Editable Tensors:**
  - `model.layers.{L}.input_layernorm.weight` ([2048])
  - `model.layers.{L}.mlp.down_proj.weight` ([2048, 6144])
  - `model.layers.{L}.mlp.gate_proj.weight` ([6144, 2048])
  - `model.layers.{L}.mlp.up_proj.weight` ([6144, 2048])
  - `model.layers.{L}.post_attention_layernorm.weight` ([2048])
  - `model.layers.{L}.self_attn.k_norm.weight` ([128])
  - `model.layers.{L}.self_attn.k_proj.weight` ([1024, 2048])
  - `model.layers.{L}.self_attn.o_proj.weight` ([2048, 2048])
  - `model.layers.{L}.self_attn.q_norm.weight` ([128])
  - `model.layers.{L}.self_attn.q_proj.weight` ([2048, 2048])
  - `model.layers.{L}.self_attn.v_proj.weight` ([1024, 2048])
- **Global Editable Tensor:** `model.norm.weight`
- **Protected Tensors (DO NOT EDIT):** `model.embed_tokens.weight`

---

## 2. Structural & Spectral Profile of Untouched Checkpoint
### Module-Type Spectral Averages (Across All 28 Layers)
| Module Suffix | Count | Mean Frobenius | Mean L_inf | Mean Cond (kappa) | Mean Effective Rank | Mean R_80% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `embed_tokens.weight` | 1 | 590.0305 | 0.3613 | 28.27 | 1992.96 | 1382.0 |
| `input_layernorm.weight` | 28 | 237.0900 | 30.2125 | N/A | N/A | N/A |
| `mlp.down_proj.weight` | 28 | 121.7005 | 0.8311 | 22.53 | 1893.55 | 1060.9 |
| `mlp.gate_proj.weight` | 28 | 132.7173 | 0.6602 | 53.28 | 1878.94 | 1028.0 |
| `mlp.up_proj.weight` | 28 | 124.5539 | 0.6111 | 43.90 | 1907.80 | 1084.3 |
| `other` | 57 | 27.1467 | 10.7897 | N/A | N/A | N/A |
| `post_attention_layernorm.weight` | 28 | 54.1566 | 9.1677 | N/A | N/A | N/A |
| `self_attn.k_proj.weight` | 28 | 46.4651 | 0.5509 | 78.44 | 834.96 | 363.6 |
| `self_attn.o_proj.weight` | 28 | 65.3931 | 0.5059 | 88406.80 | 1519.81 | 652.5 |
| `self_attn.q_proj.weight` | 28 | 73.6645 | 0.4997 | 281872.64 | 1386.48 | 510.7 |
| `self_attn.v_proj.weight` | 28 | 50.3443 | 0.2973 | 19.19 | 902.38 | 451.6 |

### Depth-Sampled Layer Spectral Profiles (Layers 0, 1, 7, 15, 23, 31, 39, 46, 47)
*(All layers 0..47 exist in the checkpoint with identical shapes; intermediate layers interpolate smoothly.)*
| Tensor Name | Shape | Frobenius | L_inf | Effective Rank | R_80% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `model.layers.0.mlp.down_proj.weight` | `[2048, 6144]` | 119.2338 | 0.6367 | 1941.4 | 1171 |
| `model.layers.0.mlp.gate_proj.weight` | `[6144, 2048]` | 138.3563 | 0.6484 | 1766.5 | 888 |
| `model.layers.0.mlp.up_proj.weight` | `[6144, 2048]` | 112.7855 | 0.4160 | 1830.1 | 965 |
| `model.layers.0.self_attn.k_proj.weight` | `[1024, 2048]` | 52.4499 | 0.8320 | 891.8 | 447 |
| `model.layers.0.self_attn.o_proj.weight` | `[2048, 2048]` | 67.1694 | 0.5156 | 1782.6 | 1007 |
| `model.layers.0.self_attn.q_proj.weight` | `[2048, 2048]` | 72.9007 | 0.6680 | 1552.0 | 686 |
| `model.layers.0.self_attn.v_proj.weight` | `[1024, 2048]` | 49.2520 | 0.2275 | 947.0 | 553 |
| `model.layers.1.mlp.down_proj.weight` | `[2048, 6144]` | 104.3635 | 0.6289 | 1915.7 | 1101 |
| `model.layers.1.mlp.gate_proj.weight` | `[6144, 2048]` | 160.7506 | 0.5273 | 1871.1 | 1011 |
| `model.layers.1.mlp.up_proj.weight` | `[6144, 2048]` | 102.4347 | 0.5234 | 1910.5 | 1088 |
| `model.layers.1.self_attn.k_proj.weight` | `[1024, 2048]` | 49.4517 | 0.5078 | 874.4 | 409 |
| `model.layers.1.self_attn.o_proj.weight` | `[2048, 2048]` | 67.2935 | 0.3711 | 1518.3 | 636 |
| `model.layers.1.self_attn.q_proj.weight` | `[2048, 2048]` | 71.5174 | 0.6094 | 1475.7 | 587 |
| `model.layers.1.self_attn.v_proj.weight` | `[1024, 2048]` | 50.6235 | 0.2617 | 919.4 | 472 |
| `model.layers.15.mlp.down_proj.weight` | `[2048, 6144]` | 116.7599 | 0.7461 | 1852.1 | 987 |
| `model.layers.15.mlp.gate_proj.weight` | `[6144, 2048]` | 116.3556 | 0.7383 | 1865.5 | 994 |
| `model.layers.15.mlp.up_proj.weight` | `[6144, 2048]` | 120.9740 | 0.5547 | 1895.3 | 1054 |
| `model.layers.15.self_attn.k_proj.weight` | `[1024, 2048]` | 42.6536 | 0.2969 | 826.1 | 344 |
| `model.layers.15.self_attn.o_proj.weight` | `[2048, 2048]` | 61.6151 | 0.4727 | 1456.1 | 583 |
| `model.layers.15.self_attn.q_proj.weight` | `[2048, 2048]` | 72.1085 | 0.4277 | 1339.4 | 450 |
| `model.layers.15.self_attn.v_proj.weight` | `[1024, 2048]` | 47.2626 | 0.4355 | 906.6 | 453 |
| `model.layers.23.mlp.down_proj.weight` | `[2048, 6144]` | 138.1814 | 1.1797 | 1876.8 | 1051 |
| `model.layers.23.mlp.gate_proj.weight` | `[6144, 2048]` | 129.7585 | 0.8555 | 1895.4 | 1059 |
| `model.layers.23.mlp.up_proj.weight` | `[6144, 2048]` | 139.0328 | 0.7852 | 1910.8 | 1089 |
| `model.layers.23.self_attn.k_proj.weight` | `[1024, 2048]` | 45.2860 | 0.3906 | 856.7 | 387 |
| `model.layers.23.self_attn.o_proj.weight` | `[2048, 2048]` | 66.8285 | 0.4004 | 1573.8 | 710 |
| `model.layers.23.self_attn.q_proj.weight` | `[2048, 2048]` | 72.3085 | 0.4531 | 1420.8 | 547 |
| `model.layers.23.self_attn.v_proj.weight` | `[1024, 2048]` | 50.4420 | 0.3438 | 909.2 | 464 |
| `model.layers.7.mlp.down_proj.weight` | `[2048, 6144]` | 113.2372 | 0.9805 | 1899.2 | 1071 |
| `model.layers.7.mlp.gate_proj.weight` | `[6144, 2048]` | 144.0195 | 0.5078 | 1885.3 | 1038 |
| `model.layers.7.mlp.up_proj.weight` | `[6144, 2048]` | 116.3777 | 0.5430 | 1922.8 | 1117 |
| `model.layers.7.self_attn.k_proj.weight` | `[1024, 2048]` | 46.2043 | 0.3340 | 856.7 | 387 |
| `model.layers.7.self_attn.o_proj.weight` | `[2048, 2048]` | 65.9064 | 0.5742 | 1519.5 | 639 |
| `model.layers.7.self_attn.q_proj.weight` | `[2048, 2048]` | 69.2299 | 0.4238 | 1439.9 | 556 |
| `model.layers.7.self_attn.v_proj.weight` | `[1024, 2048]` | 50.6287 | 0.3203 | 922.6 | 479 |
| `model.norm.weight` | `[2048]` | 102.8808 | 12.0625 | N/A | N/A |


---

## 3. Untouched Development Baseline Scores
- **Overall Unweighted Category Macro Score:** `46.81%`

| FilBench Category | Dev Macro Score (%) |
| :--- | :---: |
| Classical NLP | 81.07 |
| Cultural Knowledge | 42.21 |
| Generation | 4.40 |
| Reading Comprehension | 59.56 |

### Per-Benchmark Dev Breakdown
| Benchmark Task | Score (%) |
| :--- | :---: |
| `balita_tgl_mcf` | 88.00 |
| `belebele_ceb_mcf` | 41.11 |
| `belebele_fil_mcf` | 46.67 |
| `cebuaner_ceb_mcf` | 81.00 |
| `dengue_filipino_fil:_average` | 73.00 |
| `firecs_fil_mcf` | 50.00 |
| `global_mmlu_all_tgl_mcf:_average` | 42.11 |
| `include_tgl_mcf:_average` | 48.33 |
| `kalahi_tgl_mcf` | 60.00 |
| `newsphnli_fil_mcf` | 60.00 |
| `ntrex128_fil` | 6.95 |
| `readability_ceb_mcf` | 25.71 |
| `sib200_ceb_mcf` | 95.00 |
| `sib200_tgl_mcf` | 90.00 |
| `stingraybench_semantic_appropriateness_tgl_mcf` | 0.00 |
| `tatoeba_ceb` | 7.46 |
| `tatoeba_tgl` | 1.05 |
| `tico19_tgl` | 6.59 |
| `tlunifiedner_tgl_mcf` | 89.00 |
| `universalner_ceb_mcf` | 95.00 |
| `universalner_tgl_mcf` | 100.00 |

---

## 4. Allowed Weight-Edit Operator Algebra & Safety Budget

Every operation in `ops` computes a perturbation $\Delta W$ in `float32` using relative multiplier `delta_scale` ($\delta$), applied as $W_{\text{new}} = \text{bf16}(W + \Delta W)$:

1. `"scale_tensor"`: Whole-tensor scaling $\Delta W = \delta W$.
   - Must NOT set `row_start`, `row_end`, `col_start`, `col_end`, `rank_start`, or `rank_end`.
2. `"scale_rows"`: Contiguous row-slice scaling $\Delta W_{r_0:r_1, \dots} = \delta W_{r_0:r_1, \dots}$.
   - Requires integer `row_start` ($r_0$) and `row_end` ($r_1$) with $0 \le r_0 < r_1 \le \text{shape}[0]$.
3. `"scale_cols"`: Contiguous column-slice scaling on 2D matrices $\Delta W_{: , c_0:c_1} = \delta W_{: , c_0:c_1}$.
   - Requires integer `col_start` ($c_0$) and `col_end` ($c_1$) with $0 \le c_0 < c_1 \le \text{shape}[1]$.
4. `"spectral_band_scale"`: Truncated SVD singular-band scaling on 2D matrices:
   $$\Delta W = \sum_{i=k_0}^{k_1 - 1} \delta \, \sigma_i \, u_i v_i^\top$$
   - Requires integer `rank_start` ($k_0$) and `rank_end` ($k_1$) with $0 \le k_0 < k_1 \le \min(\text{shape}[0], \text{shape}[1])$.

### Fail-Closed Safety Budget (Strict Enforcement)
- `max_ops`: `32`
- `max_edited_tensors`: `24`
- Per-op `|delta_scale|`: `0.01 <= |delta_scale| <= 0.15` (Note: `|delta_scale| >= 0.01` is required so the edit does not underflow to zero in `bfloat16`!)
- Per-tensor relative Frobenius norm: $\|\Delta W\|_F / \|W\|_F \le 0.1$
- Per-tensor relative max-entry norm: $\|\Delta W\|_\infty / \|W\|_\infty \le 0.25$

---

## 5. Required Output Format

Return **only** a single valid JSON object inside a ```json ... ``` code block conforming to this exact schema:

```json
{
  "plan_id": "<short_snake_case_identifier>",
  "target_model": "Qwen/Qwen3-1.7B-Base",
  "hypothesis": "<concise mechanistic hypothesis explaining why these specific layers/subspaces and delta_scale signs should improve Filipino/Cebuano MCF and generation performance without collapsing fluency>",
  "ops": [
    {
      "op_type": "spectral_band_scale",
      "tensor_name": "model.layers.40.self_attn.o_proj.weight",
      "delta_scale": -0.05,
      "rank_start": 0,
      "rank_end": 8,
      "rationale": "<why this specific tensor, band/slice, and sign were chosen>"
    }
  ]
}
```
