# NL2SH: Improving Small LLMs for Natural Language to Bash Translation

[中文版](README_CN.md)

Based on the NAACL 2025 paper [LLM-Supported Natural Language to Bash Translation](https://aclanthology.org/2025.naacl-long.555/). This project reproduces and improves the NL2SH performance of small-parameter models (0.5B-3B).

## Goal

Reproduce the Base / Parse / ICL / IWL metrics from the paper for the Qwen2.5-Coder model family, and improve small model performance through better inference strategies and training methods.

## Current Results

| Model | Base | Parse | ICL (completion) | ICL (chat) | Paper ICL |
|-------|------|-------|-------------------|------------|-----------|
| qwen2.5-coder-0.5b-instruct | 0.110 | 0.337 | - | 0.000 | 0.36 |
| qwen2.5-coder-1.5b-instruct | 0.220 | 0.497 | **0.503** | - | 0.44 |
| qwen2.5-coder-3b-instruct | 0.260 | 0.583 | **0.597** | - | 0.50 |

> Evaluator: InterCode-ALFA (icalfa), exec + mxbai-embed cosine similarity >= 0.75

## Key Finding: ICL Inference Method Matters for Small Models

The paper uses chat template for ICL inference, but the 0.5B model fails under chat template — it regurgitates ICL examples instead of generating commands for the test query. Switching to raw completion (plain text continuation without chat template) significantly improves ICL scores for 1.5B and 3B models, surpassing the paper's reported values.

- **chat template**: `apply_chat_template` -> system role + user role -> `model.generate()`
- **raw completion**: plain text prompt -> `model.generate()` -> extract first line of output

## Quick Start

### Environment Setup

```bash
# Dependencies
pip install torch transformers datasets tqdm icalfa accelerate

# Docker (required for icalfa evaluation)
sudo usermod -aG docker $USER
sg docker bash

# Build icalfa Docker images (first time, ~5-10 min)
bash redili_Reproduce/build_icalfa_images.sh

# Pull embedding model (required for icalfa evaluation)
ollama pull mxbai-embed-large
```

### Download Models

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir ~/models/qwen2.5-coder-0.5b-instruct
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ~/models/qwen2.5-coder-1.5b-instruct
hf download Qwen/Qwen2.5-Coder-3B-Instruct   --local-dir ~/models/qwen2.5-coder-3b-instruct
```

### Run Evaluation

```bash
cd redili_Reproduce

# Select ICL examples (first time, ~3 min)
python 0_precompute_icl.py

# Run Base / Parse / ICL (ICL defaults to completion mode)
python 1_reproduce_no_train.py --models ~/models/qwen2.5-coder-1.5b-instruct

# Run ICL with chat template (aligned with paper; 0.5B will fail)
python 1_reproduce_no_train.py --models ~/models/qwen2.5-coder-0.5b-instruct --icl_mode chat --force

# View summary of all results
python summarize_results.py
```

## Project Structure

```
NL2SH/
├── paper/                          # Paper LaTeX source
├── code/                           # Original paper code (notebooks)
├── nl2sh_alfa/                     # Dataset (local CSV)
│   ├── train.csv                   # Training set (40,939 examples)
│   └── test.csv                    # Test set (300 examples)
├── redili_Reproduce/               # Reproduction and improvement code
│   ├── 0_precompute_icl.py         # ICL example selection (mxbai-embed + k-means)
│   ├── 1_reproduce_no_train.py     # Base / Parse / ICL evaluation
│   ├── 2_train_and_eval_iwl.py     # SFT training + IWL evaluation
│   ├── build_icalfa_images.sh      # Build Docker evaluation images
│   ├── icl_examples.json           # 25 ICL examples
│   ├── summarize_results.py        # Results summary script
│   ├── results/                    # Evaluation result CSVs
│   └── PLAN.md                     # Reproduction and improvement plan
└── AGENTS.md
```

## Evaluation Metrics

| Metric | Prompt | Post-processing | Model |
|--------|--------|-----------------|-------|
| Base | `prompt-baseline` (with "no markdown" constraint) | None | Original Instruct model |
| Parse | `prompt-other` | `parse_bash()` strips markdown | Original Instruct model |
| ICL | `prompt-other` + 25 few-shot examples | `parse_bash_completion()` | Original Instruct model |
| IWL | `prompt-other` | None | SFT fine-tuned model |

## Improvement Directions

1. **ICL inference optimization**: raw completion instead of chat template for better few-shot performance
2. **LoRA fine-tuning**: use smaller rank (r=8-16) to avoid overfitting on small models
3. **Training strategy**: reduce epochs (3-5 vs paper's 10), add eval loss early stopping
4. **Data filtering**: use high-quality subsets of the training set instead of all 40k examples

## Citation

```bibtex
@inproceedings{westenfelder-etal-2025-llm,
    title = "{LLM}-Supported Natural Language to Bash Translation",
    author = "Westenfelder, Finnian and Hemberg, Erik and Moskal, Stephen and O'Reilly, Una-May and Chiricescu, Silviu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.555/",
    pages = "11135--11147",
}
```

## Acknowledgements

- Original paper: [LLM-Supported Natural Language to Bash Translation](https://aclanthology.org/2025.naacl-long.555/) (NAACL 2025)
- Evaluator: [InterCode-ALFA](https://github.com/westenfelder/InterCode-ALFA)
- Dataset: [NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
