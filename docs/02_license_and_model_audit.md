# Model, Benchmark, and Dataset License Audit

## Candidate target models

| Candidate | Base HF ID | Instruct variant | Params | License | Gated on HF? |
|---|---|---|---:|---|---|
| Sailor2 1B | `sail/Sailor2-1B` | `sail/Sailor2-1B-Chat` | 0.99B | Apache-2.0 | No |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-1.7B` | 1.72B | Apache-2.0 | No |
| Gemma 3 1B | `google/gemma-3-1b-pt` | `google/gemma-3-1b-it` | 1.00B | Gemma Terms of Use | Yes |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | 1.24B | Llama 3.2 Community License | Yes |

Model-specific license URLs and redistribution notes are frozen beside each
candidate in `configs/models/`. Verify the current upstream terms before
redistributing weights, deltas, or edited checkpoints.

## Repository policy

No model weights (`.safetensors`, `.bin`) are committed to Git. Checkpoints
remain exclusively in the git-ignored `checkpoints/` directory.

## Upstream evaluation repositories

The FilBench evaluation repository is not vendored. If used, clone it locally
under the git-ignored `external/filbench-eval/` directory and record its exact
commit in the experiment manifest.

## Benchmark datasets

FilBench datasets retain their upstream licenses. Do not commit dataset
examples, cached Parquet/Arrow files, or test questions; keep detailed outputs
inside git-ignored `experiments/<EXP_ID>/results/`.
