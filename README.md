# FilBench-WSP

**LLM-Driven Weight-Space Programming for Philippine Language Capability Enhancement**

An experimental research project investigating whether an external LLM can directly modify the weights of smaller open-weight language models to improve their performance on Filipino language benchmarks.

## Research Objective

Explore whether direct weight modifications can improve a model's Filipino language capabilities without conventional fine-tuning.

## Experiment Overview

We will:

1. Evaluate an original open-weight model using FilBench.
2. Use an external LLM to propose direct weight modifications.
3. Evaluate the modified model using the same benchmark.
4. Compare results against the original model and controlled random edits.
5. Analyze performance improvements, regressions, and limitations.

**Core operation:**

\[
\theta_{\text{edited}} = \theta_{\text{base}} + \Delta\theta
\]

## Initial Tools

- Python
- PyTorch
- Hugging Face Transformers
- Safetensors
- FilBench
- Lighteval

## Candidate Models

- Gemma 3 1B
- Llama 3.2 1B
- Qwen3 1.7B
- Sailor2 1B

Final model selection will depend on compatibility, licensing, tokenizer fertility,
and available compute. Candidate metadata is stored in `configs/models/`.

## Repository Structure

```text
filbench-wsp/
├── docs/           # Research documentation and audits
├── configs/        # Frozen experiment configurations
├── src/filbench_wsp/
├── experiments/    # Reproducible experiment records and outputs
├── checkpoints/    # Local model weights (never committed)
├── external/       # Isolated upstream evaluation suite
├── tests/          # Automated safety and validity checks
├── scripts/        # Step-by-step CLI entry points
├── README.md
└── LICENSE
```

## Setup

Python 3.10+ is supported. Create an environment and install the project
with its development dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

The Phase 1 workflow is intentionally staged:

```bash
python scripts/01_inspect_candidates.py
python scripts/02_prepare_splits.py --output configs/dev_split_manifest.json
python scripts/03_run_eval.py --help
python scripts/04_score_local.py --help
```

Model checkpoints, upstream evaluation code, and experiment outputs are local
artifacts and are excluded from version control.

## Collaboration

This is a shared research project. Both researchers contribute to experimentation, implementation, analysis, and writing.

We prioritize reproducibility, open discussion, and collaborative review.

## Reproducibility

Experiments should document:

- Model checkpoints and hashes
- Editing configurations
- Random seeds
- Evaluation settings
- Weight modification logs
- Results and analysis

Large model checkpoints and datasets should not be committed to this repository.

## License

The project's original code and documentation will use their respective licenses. Third-party models, datasets, and tools remain subject to their original terms.

## Research Status

**Currently in development.** No experimental results are claimed yet.
