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

Final model selection will depend on compatibility, licensing, and available compute.

## Repository Structure

```text
filbench-wsp/
├── docs/           # Research documentation
├── configs/        # Experiment configurations
├── src/            # Research implementation
├── experiments/    # Baselines, edits, and controls
├── results/        # Evaluation results
├── notebooks/      # Exploratory analysis
├── tests/          # Tests
├── scripts/        # Utility scripts
├── README.md
└── LICENSE
```

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
