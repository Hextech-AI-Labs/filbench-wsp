# Direct Weight-Space Intervention Task (FilBench-WSP)
Packet ID: `packet_v1_dev_zero_shot`

You are participating in a scientific experiment testing whether a frontier LLM can propose a **direct, training-free weight perturbation** ($\theta_{\text{edited}} = \theta_{\text{base}} + \Delta\theta$) to improve the Filipino/Tagalog/Cebuano performance of `sail/Sailor2-1B` on the FilBench development split, without gradient descent, fine-tuning, or adapters.

---

## 1. Target Model Architecture (`sail/Sailor2-1B`)
- **Architecture:** `Qwen2ForCausalLM` (`bfloat16`)
- **Depth & Width:** `48` transformer layers (`model.layers.0` .. `model.layers.47`), `hidden_size = 896`, `intermediate_size = 4864`
- **Grouped-Query Attention (GQA):**
  - `14` Query heads (`head_dim = 64`, total Q/O dim = `896`)
  - `2` Key/Value heads (`head_dim = 64`, total K/V dim = `128`)
  - Query heads `0..6` (rows/cols `[0, 448)`) share KV head `0` (rows `[0, 64)`).
  - Query heads `7..13` (rows/cols `[448, 896)`) share KV head `1` (rows `[64, 128)`).
- **Per-Layer Editable Tensors (for `L` in `0..47`):**
  - `model.layers.{L}.input_layernorm.weight` (`[896]`)
  - `model.layers.{L}.self_attn.q_proj.weight` (`[896, 896]`) and `.bias` (`[896]`)
  - `model.layers.{L}.self_attn.k_proj.weight` (`[128, 896]`) and `.bias` (`[128]`)
  - `model.layers.{L}.self_attn.v_proj.weight` (`[128, 896]`) and `.bias` (`[128]`)
  - `model.layers.{L}.self_attn.o_proj.weight` (`[896, 896]`)
  - `model.layers.{L}.post_attention_layernorm.weight` (`[896]`)
  - `model.layers.{L}.mlp.gate_proj.weight` (`[4864, 896]`)
  - `model.layers.{L}.mlp.up_proj.weight` (`[4864, 896]`)
  - `model.layers.{L}.mlp.down_proj.weight` (`[896, 4864]`)
- **Global Editable Tensor:** `model.norm.weight` (`[896]`)
- **Protected Tensors (DO NOT EDIT):** `model.embed_tokens.weight`, `lm_head.weight`

---

## 2. Structural & Spectral Profile of Untouched Checkpoint
### Module-Type Spectral Averages (Across All 48 Layers)
| Module Suffix | Count | Mean Frobenius | Mean L_inf | Mean Cond (kappa) | Mean Effective Rank | Mean R_80% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `embed_tokens.weight` | 1 | 62.9771 | 0.0762 | 18.15 | 865.68 | 597.0 |
| `input_layernorm.weight` | 48 | 47.7555 | 5.6817 | N/A | N/A | N/A |
| `lm_head.weight` | 1 | 88.0017 | 0.0742 | 10.24 | 875.01 | 619.0 |
| `mlp.down_proj.weight` | 48 | 19.3189 | 0.3201 | 11.64 | 846.88 | 512.2 |
| `mlp.gate_proj.weight` | 48 | 22.5731 | 0.2250 | 24.96 | 844.84 | 506.6 |
| `mlp.up_proj.weight` | 48 | 19.5559 | 0.2186 | 10.40 | 862.35 | 542.8 |
| `other` | 1 | 238.5535 | 16.8750 | N/A | N/A | N/A |
| `post_attention_layernorm.weight` | 48 | 50.6417 | 3.8747 | N/A | N/A | N/A |
| `self_attn.k_proj.bias` | 48 | 52.5750 | 27.9303 | N/A | N/A | N/A |
| `self_attn.k_proj.weight` | 48 | 6.0038 | 0.1970 | 28.10 | 108.02 | 51.8 |
| `self_attn.o_proj.weight` | 48 | 7.8206 | 0.1719 | 19259.61 | 624.79 | 244.2 |
| `self_attn.q_proj.bias` | 48 | 65.0985 | 19.1458 | N/A | N/A | N/A |
| `self_attn.q_proj.weight` | 48 | 9.4315 | 0.2056 | 67822.32 | 565.85 | 182.5 |
| `self_attn.v_proj.bias` | 48 | 3.5509 | 1.9369 | N/A | N/A | N/A |
| `self_attn.v_proj.weight` | 48 | 3.6211 | 0.0830 | 5.95 | 123.39 | 80.1 |

### Depth-Sampled Layer Spectral Profiles (Layers 0, 1, 7, 15, 23, 31, 39, 46, 47)
*(All layers 0..47 exist in the checkpoint with identical shapes; intermediate layers interpolate smoothly.)*
| Tensor Name | Shape | Frobenius | L_inf | Effective Rank | R_80% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `model.layers.0.mlp.down_proj.weight` | `[896, 4864]` | 22.5071 | 0.2119 | 874.6 | 587 |
| `model.layers.0.mlp.gate_proj.weight` | `[4864, 896]` | 23.4348 | 0.2695 | 817.5 | 471 |
| `model.layers.0.mlp.up_proj.weight` | `[4864, 896]` | 21.5377 | 0.1611 | 879.6 | 600 |
| `model.layers.0.self_attn.k_proj.weight` | `[128, 896]` | 9.0613 | 0.6914 | 66.9 | 13 |
| `model.layers.0.self_attn.o_proj.weight` | `[896, 896]` | 7.1702 | 0.5312 | 453.3 | 79 |
| `model.layers.0.self_attn.q_proj.weight` | `[896, 896]` | 17.7095 | 0.5586 | 287.4 | 23 |
| `model.layers.0.self_attn.v_proj.weight` | `[128, 896]` | 3.4296 | 0.0889 | 119.0 | 68 |
| `model.layers.1.mlp.down_proj.weight` | `[896, 4864]` | 15.5053 | 0.1318 | 864.9 | 554 |
| `model.layers.1.mlp.gate_proj.weight` | `[4864, 896]` | 18.7291 | 0.1338 | 849.1 | 518 |
| `model.layers.1.mlp.up_proj.weight` | `[4864, 896]` | 17.0508 | 0.0830 | 864.3 | 545 |
| `model.layers.1.self_attn.k_proj.weight` | `[128, 896]` | 7.7018 | 0.2891 | 67.9 | 13 |
| `model.layers.1.self_attn.o_proj.weight` | `[896, 896]` | 4.5450 | 0.1289 | 559.4 | 160 |
| `model.layers.1.self_attn.q_proj.weight` | `[896, 896]` | 16.7667 | 0.3457 | 294.5 | 24 |
| `model.layers.1.self_attn.v_proj.weight` | `[128, 896]` | 2.5851 | 0.0562 | 119.8 | 69 |
| `model.layers.15.mlp.down_proj.weight` | `[896, 4864]` | 17.4483 | 0.2197 | 854.2 | 523 |
| `model.layers.15.mlp.gate_proj.weight` | `[4864, 896]` | 20.5338 | 0.1631 | 852.9 | 526 |
| `model.layers.15.mlp.up_proj.weight` | `[4864, 896]` | 18.3106 | 0.2070 | 867.0 | 555 |
| `model.layers.15.self_attn.k_proj.weight` | `[128, 896]` | 5.9338 | 0.2090 | 112.1 | 57 |
| `model.layers.15.self_attn.o_proj.weight` | `[896, 896]` | 7.2205 | 0.1328 | 674.5 | 291 |
| `model.layers.15.self_attn.q_proj.weight` | `[896, 896]` | 8.6838 | 0.1475 | 609.7 | 211 |
| `model.layers.15.self_attn.v_proj.weight` | `[128, 896]` | 3.3704 | 0.0508 | 125.7 | 86 |
| `model.layers.23.mlp.down_proj.weight` | `[896, 4864]` | 16.6305 | 0.1553 | 829.4 | 473 |
| `model.layers.23.mlp.gate_proj.weight` | `[4864, 896]` | 18.8171 | 0.2070 | 846.0 | 504 |
| `model.layers.23.mlp.up_proj.weight` | `[4864, 896]` | 18.0000 | 0.2910 | 859.5 | 532 |
| `model.layers.23.self_attn.k_proj.weight` | `[128, 896]` | 5.3538 | 0.1338 | 106.2 | 51 |
| `model.layers.23.self_attn.o_proj.weight` | `[896, 896]` | 6.5659 | 0.0850 | 627.5 | 238 |
| `model.layers.23.self_attn.q_proj.weight` | `[896, 896]` | 8.2620 | 0.1484 | 535.3 | 161 |
| `model.layers.23.self_attn.v_proj.weight` | `[128, 896]` | 3.4611 | 0.0703 | 123.3 | 78 |
| `model.layers.31.mlp.down_proj.weight` | `[896, 4864]` | 17.6178 | 0.2539 | 847.3 | 510 |
| `model.layers.31.mlp.gate_proj.weight` | `[4864, 896]` | 20.0391 | 0.1973 | 851.2 | 518 |
| `model.layers.31.mlp.up_proj.weight` | `[4864, 896]` | 18.6958 | 0.3047 | 865.2 | 548 |
| `model.layers.31.self_attn.k_proj.weight` | `[128, 896]` | 5.2172 | 0.1621 | 109.9 | 52 |
| `model.layers.31.self_attn.o_proj.weight` | `[896, 896]` | 6.4732 | 0.0996 | 662.0 | 276 |
| `model.layers.31.self_attn.q_proj.weight` | `[896, 896]` | 7.8412 | 0.1416 | 602.4 | 205 |
| `model.layers.31.self_attn.v_proj.weight` | `[128, 896]` | 3.1713 | 0.0513 | 123.6 | 79 |
| `model.layers.39.mlp.down_proj.weight` | `[896, 4864]` | 18.3720 | 0.2910 | 855.3 | 526 |
| `model.layers.39.mlp.gate_proj.weight` | `[4864, 896]` | 22.3636 | 0.1895 | 849.5 | 517 |
| `model.layers.39.mlp.up_proj.weight` | `[4864, 896]` | 19.0501 | 0.2305 | 861.6 | 541 |
| `model.layers.39.self_attn.k_proj.weight` | `[128, 896]` | 4.5345 | 0.1504 | 113.8 | 57 |
| `model.layers.39.self_attn.o_proj.weight` | `[896, 896]` | 6.8859 | 0.1406 | 681.9 | 295 |
| `model.layers.39.self_attn.q_proj.weight` | `[896, 896]` | 7.5973 | 0.1445 | 631.1 | 236 |
| `model.layers.39.self_attn.v_proj.weight` | `[128, 896]` | 4.0798 | 0.0703 | 124.7 | 83 |
| `model.layers.46.mlp.down_proj.weight` | `[896, 4864]` | 21.7285 | 0.4141 | 860.8 | 538 |
| `model.layers.46.mlp.gate_proj.weight` | `[4864, 896]` | 25.2324 | 0.2168 | 850.6 | 527 |
| `model.layers.46.mlp.up_proj.weight` | `[4864, 896]` | 22.5104 | 0.1602 | 857.0 | 535 |
| `model.layers.46.self_attn.k_proj.weight` | `[128, 896]` | 4.5689 | 0.2109 | 102.3 | 47 |
| `model.layers.46.self_attn.o_proj.weight` | `[896, 896]` | 9.4657 | 0.5195 | 612.1 | 242 |
| `model.layers.46.self_attn.q_proj.weight` | `[896, 896]` | 8.6635 | 0.1060 | 540.0 | 172 |
| `model.layers.46.self_attn.v_proj.weight` | `[128, 896]` | 4.2943 | 0.1079 | 109.9 | 55 |
| `model.layers.47.mlp.down_proj.weight` | `[896, 4864]` | 18.9811 | 0.3574 | 856.5 | 530 |
| `model.layers.47.mlp.gate_proj.weight` | `[4864, 896]` | 21.1572 | 0.1235 | 849.7 | 519 |
| `model.layers.47.mlp.up_proj.weight` | `[4864, 896]` | 20.5945 | 0.2109 | 860.0 | 538 |
| `model.layers.47.self_attn.k_proj.weight` | `[128, 896]` | 3.2344 | 0.1050 | 109.1 | 50 |
| `model.layers.47.self_attn.o_proj.weight` | `[896, 896]` | 6.9354 | 0.0559 | 634.4 | 244 |
| `model.layers.47.self_attn.q_proj.weight` | `[896, 896]` | 6.5528 | 0.1523 | 583.2 | 192 |
| `model.layers.47.self_attn.v_proj.weight` | `[128, 896]` | 5.4119 | 0.0928 | 120.9 | 72 |
| `model.layers.7.mlp.down_proj.weight` | `[896, 4864]` | 17.1503 | 0.3887 | 858.2 | 536 |
| `model.layers.7.mlp.gate_proj.weight` | `[4864, 896]` | 21.1499 | 0.1875 | 860.9 | 544 |
| `model.layers.7.mlp.up_proj.weight` | `[4864, 896]` | 17.5837 | 0.1475 | 869.4 | 564 |
| `model.layers.7.self_attn.k_proj.weight` | `[128, 896]` | 5.3981 | 0.1562 | 114.1 | 60 |
| `model.layers.7.self_attn.o_proj.weight` | `[896, 896]` | 7.0569 | 0.1748 | 650.1 | 262 |
| `model.layers.7.self_attn.q_proj.weight` | `[896, 896]` | 8.0081 | 0.1396 | 584.8 | 193 |
| `model.layers.7.self_attn.v_proj.weight` | `[128, 896]` | 3.6102 | 0.0791 | 124.7 | 84 |
| `model.norm.weight` | `[896]` | 238.5535 | 16.8750 | N/A | N/A |


---

## 3. Untouched Development Baseline Scores (`n = 1,431` dev examples)
- **Overall Unweighted Category Macro Score:** `31.75%`

| FilBench Category | Dev Macro Score (%) |
| :--- | :---: |
| Classical NLP | 35.43 |
| Cultural Knowledge | 23.25 |
| Generation | 16.97 |
| Reading Comprehension | 51.35 |

### Per-Benchmark Dev Breakdown
| Benchmark Task | Score (%) |
| :--- | :---: |
| `balita_tgl_mcf` | 34.00 |
| `belebele_ceb_mcf` | 23.33 |
| `belebele_fil_mcf` | 22.22 |
| `cebuaner_ceb_mcf` | 5.00 |
| `dengue_filipino_fil:_average` | 74.00 |
| `firecs_fil_mcf` | 25.00 |
| `global_mmlu_all_tgl_mcf:_average` | 22.81 |
| `include_tgl_mcf:_average` | 29.94 |
| `kalahi_tgl_mcf` | 25.00 |
| `newsphnli_fil_mcf` | 52.00 |
| `ntrex128_fil` | 15.40 |
| `readability_ceb_mcf` | 31.43 |
| `sib200_ceb_mcf` | 20.00 |
| `sib200_tgl_mcf` | 5.00 |
| `stingraybench_semantic_appropriateness_tgl_mcf` | 50.00 |
| `tatoeba_ceb` | 37.36 |
| `tatoeba_tgl` | 16.71 |
| `tico19_tgl` | 12.93 |
| `tlunifiedner_tgl_mcf` | 47.00 |
| `universalner_ceb_mcf` | 10.00 |
| `universalner_tgl_mcf` | 25.00 |

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
  "target_model": "sail/Sailor2-1B",
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
