# RetroBug Replication Package

Replication artifact for **RetroBug** — an agentic backward case-based reasoning (CBR) system for binary bug-fix commit classification (Bug-Fix vs. Non-Bug-Fix).

This package contains experiment code, evaluation scripts, and precomputed results for RQ1–RQ3. All methods are evaluated under the same **5-fold stratified cross-validation** protocol on **798 commits**.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Data Preparation](#data-preparation)
7. [Running the Experiments](#running-the-experiments)
8. [Outputs and Results](#outputs-and-results)
9. [Verifying Paper Numbers](#verifying-paper-numbers)
10. [Verification Checklist](#verification-checklist)
11. [Troubleshooting](#troubleshooting)

---

## Overview

| Component | Description |
|-----------|-------------|
| **RetroBug (ours)** | Multi-step agent with KB retrieval, backward CBR, and 6 analysis tools; uses Claude Sonnet 4.5 by default |
| **Classical baselines** | Keyword heuristic, TF-IDF (LR/RF/XGB), hybrid features (LR/RF/XGB), kNN retrieval |
| **LLM baselines** | Zero-shot Claude, GPT-4o, and CodeLlama-13b (masked message + git diff) |
| **Evaluation** | Per-class P/R/F1, accuracy, MCC; 798 commits, 5 folds |

**Research questions (replication scope):**

- **RQ1:** RetroBug vs. classical baselines (full 5-fold CV)
- **RQ2:** RetroBug vs. zero-shot LLM baselines (fold 0; same test split)
- **RQ3:** Cross-fold stability and per-class balance (RetroBug only)

> **Note:** An ablation study is maintained separately and is **not included** in this package.

---

## Repository Layout

```
RetroBug/
├── README.md                    ← you are here
├── scripts/package_replication.sh
├── agentic_experiment/          ← experiment code + results
│   ├── requirements.txt
│   ├── .env.example             ← copy to .env
│   ├── run_experiment.py        ← RetroBug agentic pipeline
│   ├── run_baselines.py         ← all baselines
│   ├── summarize_baselines.py   ← aggregate comparison report
│   ├── agent.py, tools.py, knowledge_base.py, …
│   └── data/
│       ├── merged_dataset.jsonl
│       ├── cv_splits.json
│       ├── fold_{0..4}/         ← RetroBug per-fold outputs
│       ├── baselines/           ← baseline per-fold outputs
│       └── cross_fold_summary.txt
├── bugfix_pipeline/data/        ← source CSV datasets
│   ├── dataset1_100fix_300nonfix.csv
│   └── dataset2_100fix_300nonfix.csv
└── claude_message_experiment/   ← shared prompt/parser for LLM baselines
    ├── prompts.py
    └── parse_response.py
```

---

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| **Python 3.10+** | All scripts |
| **Anthropic API key** | RetroBug agent + Claude baselines |
| **OpenAI API key** | GPT-4o baseline only (optional) |
| **NVIDIA GPU (~26 GB VRAM)** | CodeLlama-13b baseline only (optional) |
| **Internet access** | API calls + downloading `all-MiniLM-L6-v2` embeddings on first run |

RetroBug and classical baselines do **not** require a GPU.

---

## Installation

```bash
cd agentic_experiment
python3 -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys (see Configuration below)
```

All experiment commands below assume your working directory is `agentic_experiment/`.

---

## Configuration

Create `agentic_experiment/.env` from `.env.example`:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes (agent + Claude baselines) | — | Anthropic API authentication |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-5` | Model for RetroBug and Claude baselines |
| `OPENAI_API_KEY` | GPT-4o baseline only | — | OpenAI API authentication |
| `OPENAI_MODEL` | No | `gpt-4o` | GPT-4o baseline model |
| `LLM_BACKEND` | No | `claude` | Agent backend: `claude` or `codellama` |
| `CODELLAMA_PATH` | CodeLlama runs | — | Path to `CodeLlama-13b-Instruct-hf` |
| `CUDA_VISIBLE_DEVICES` | CodeLlama runs | — | GPU device ID |

Key hyperparameters are in `agentic_experiment/config.py` (5 CV folds, seed=42; max 12 agent steps; `all-MiniLM-L6-v2` embeddings).

---

## Data Preparation

Source CSVs must exist at `bugfix_pipeline/data/dataset{1,2}_100fix_300nonfix.csv`.

```bash
cd agentic_experiment
python3 run_experiment.py --prepare-data
```

This writes `data/merged_dataset.jsonl` (798 commits) and `data/cv_splits.json`. Precomputed data and results are already bundled under `agentic_experiment/data/`.

---

## Running the Experiments

### RetroBug (agentic pipeline)

```bash
cd agentic_experiment
python3 run_experiment.py              # full 5-fold (resumable)
python3 run_experiment.py --fold 0     # single fold
python3 run_experiment.py --metrics-only   # recompute metrics only
```

### Classical baselines (no API)

```bash
python3 run_baselines.py --baseline all_classical
python3 run_baselines.py --baseline keyword --fold 0
```

### LLM baselines (fold 0 for RQ2)

```bash
python3 run_baselines.py --baseline claude_message_diff --fold 0
python3 run_baselines.py --baseline gpt4o_message_diff --fold 0
python3 run_baselines.py --baseline codellama_message_diff --fold 0
```

### Summarize and compare

```bash
python3 run_experiment.py --metrics-only   # → data/cross_fold_summary.txt
python3 summarize_baselines.py             # → data/baselines/baselines_comparison.txt
```

---

## Outputs and Results

| Method | BF F1 | Acc. | MCC | Scope |
|--------|-------|------|-----|-------|
| **RetroBug** | **0.778** | **0.871** | **0.703** | 5-fold, n=798 |
| TF-IDF + LR (msg) | 0.500 | 0.732 | 0.319 | 5-fold, n=798 |
| kNN retrieval | 0.423 | 0.743 | 0.264 | 5-fold, n=798 |
| GPT-4o zero-shot | 0.629 | 0.794 | 0.493 | fold 0, n=160 |
| Claude zero-shot | 0.598 | 0.731 | 0.446 | fold 0, n=160 |
| CodeLlama-13b | 0.431 | 0.585 | 0.171 | fold 0, n=159* |

\*One CodeLlama response was unparseable and excluded from metrics.

Per-fold outputs live under `agentic_experiment/data/fold_{0..4}/` and `agentic_experiment/data/baselines/<name>/fold_*/`.

---

## Verifying Paper Numbers

```bash
cd agentic_experiment
python3 run_experiment.py --metrics-only
python3 summarize_baselines.py
```

Expected RetroBug pooled BF F1 ≈ 0.778. See `data/cross_fold_summary.txt` and `data/baselines/baselines_comparison.txt`.

---

## Verification Checklist

- [ ] `python3 run_experiment.py --prepare-data` produces 798 commits in `merged_dataset.jsonl`
- [ ] `python3 run_experiment.py --metrics-only` reports RetroBug BF F1 ≈ **0.778 ± 0.027**
- [ ] `python3 summarize_baselines.py` shows TF-IDF + LR BF F1 ≈ **0.500**
- [ ] `data/fold_0/metrics.txt` shows RetroBug BF F1 ≈ **0.776**, MCC ≈ **0.698**
- [ ] `data/baselines/gpt4o_message_diff/fold_0/metrics.txt` shows BF F1 ≈ **0.629**

---

## Troubleshooting

**Missing source CSVs** — ensure `bugfix_pipeline/data/dataset{1,2}_100fix_300nonfix.csv` exist.

**`ModuleNotFoundError: claude_message_experiment`** — the sibling `claude_message_experiment/` directory must be present.

**API authentication errors** — check `agentic_experiment/.env` for valid keys.

**Interrupted runs** — both pipelines default to `--resume`. Use `--no-resume` to overwrite a fold.

---

## Citation

```bibtex
@inproceedings{retrobug2026,
  title   = {RetroBug: Agentic Backward Case-Based Reasoning for Bug-Fix Commit Classification},
  author  = {...},
  booktitle = {...},
  year    = {2026}
}
```

Do not commit `.env` files containing API keys.
