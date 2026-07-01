# RetroBug Replication Package

Replication artifact for **RetroBug** — an agentic backward case-based reasoning (CBR) system for binary bug-fix commit classification (Bug-Fix vs. Non-Bug-Fix).

This package contains the full experiment code, evaluation scripts, precomputed results, and paper tables used in our study. All methods are evaluated under the same **5-fold stratified cross-validation** protocol on **798 commits**.

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
9. [Reproducing Paper Tables](#reproducing-paper-tables)
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

**Research questions addressed (replication package scope):**

- **RQ1:** RetroBug vs. classical baselines (full 5-fold CV)
- **RQ2:** RetroBug vs. zero-shot LLM baselines (fold 0; same test split)
- **RQ3:** Cross-fold stability and per-class balance (RetroBug only)

> **Note:** An ablation study (component analysis) is maintained separately and is
> **not included** in this replication package. See `../REPLICATION.md` for packaging
> details.

---

## Repository Layout

This experiment lives inside the parent `BugFix/` repository. Three directories are required:

```
BugFix/
├── agentic_experiment/          ← THIS PACKAGE (main entry point)
│   ├── README.md                ← you are here
│   ├── requirements.txt
│   ├── .env.example             ← copy to .env
│   ├── config.py                ← paths, hyperparameters, agent prompt
│   ├── run_experiment.py        ← RetroBug agentic pipeline
│   ├── run_baselines.py         ← all baselines
│   ├── summarize_baselines.py   ← aggregate comparison report
│   ├── load_data.py             ← merge datasets + CV splits
│   ├── compute_metrics.py       ← metrics computation
│   ├── agent.py                 ← autonomous CBR agent
│   ├── knowledge_base.py        ← sentence-transformer KB
│   ├── tools.py                 ← agent tools (6 tools)
│   ├── local_llm.py             ← LLM backend router
│   ├── claude_llm.py            ← Anthropic API backend
│   ├── codellama_llm.py         ← local CodeLlama backend
│   ├── baselines/
│   │   ├── registry.py          ← baseline names and factory
│   │   ├── classical.py         ← keyword, TF-IDF, hybrid, kNN
│   │   ├── llm_baselines.py     ← Claude, GPT-4o, CodeLlama
│   │   ├── features.py          ← hand-crafted hybrid features
│   │   └── gpt_prompts.py       ← GPT-4o CCS-style prompt
│   ├── data/                    ← datasets, splits, all run outputs
│   │   ├── merged_dataset.jsonl
│   │   ├── cv_splits.json
│   │   ├── fold_{0..4}/         ← RetroBug per-fold outputs
│   │   ├── baselines/           ← baseline per-fold outputs
│   │   └── cross_fold_summary.txt
│   └── paper_tables/            ← LaTeX tables and results prose
│       ├── rq1_results.tex
│       ├── rq2_results.tex
│       ├── rq3_results.tex
│       ├── baselines.tex
│       ├── confusion_matrix.tex
│       └── generate_per_fold_tables.py
│
├── bugfix_pipeline/data/        ← SOURCE DATA (required)
│   ├── dataset1_100fix_300nonfix.csv
│   └── dataset2_100fix_300nonfix.csv
│
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

RetroBug and classical baselines do **not** require a GPU. The knowledge-base embedding model (`all-MiniLM-L6-v2`) runs on CPU.

---

## Installation

```bash
cd agentic_experiment
python3 -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys (see Configuration below)
```

All commands below assume your working directory is `agentic_experiment/`.

---

## Configuration

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes (agent + Claude baselines) | — | Anthropic API authentication |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-5` | Model for RetroBug and Claude baselines |
| `OPENAI_API_KEY` | GPT-4o baseline only | — | OpenAI API authentication |
| `OPENAI_MODEL` | No | `gpt-4o` | GPT-4o baseline model |
| `LLM_BACKEND` | No | `claude` | Agent backend: `claude` or `codellama` |
| `CODELLAMA_PATH` | CodeLlama runs | — | Path to `CodeLlama-13b-Instruct-hf` |
| `CUDA_VISIBLE_DEVICES` | CodeLlama runs | — | GPU device ID |

Key hyperparameters are in `config.py`:

| Parameter | Value |
|-----------|-------|
| CV folds | 5 (stratified, seed=42) |
| Max agent steps | 12 |
| Max KB searches | 2 |
| Diff character limit | 12,000 |
| KB top-K | 1 |
| Embedding model | `all-MiniLM-L6-v2` |

---

## Data Preparation

Source CSVs must exist at:

```
../bugfix_pipeline/data/dataset1_100fix_300nonfix.csv
../bugfix_pipeline/data/dataset2_100fix_300nonfix.csv
```

Each CSV contains human-annotated commits with: `sha`, `commit_message`, `masked_commit_message`, `git_diff`, `human_label`, `commit_url`, etc.

Merge datasets and create CV splits:

```bash
python3 run_experiment.py --prepare-data
# equivalent:
python3 load_data.py
```

This writes:

- `data/merged_dataset.jsonl` — 798 deduplicated commits
- `data/cv_splits.json` — fold assignments (seed=42)

**Fold sizes:**

| Folds | Train | Test |
|-------|-------|------|
| 0, 1, 2 | 638 | 160 |
| 3, 4 | 639 | 159 |

> **Note:** Precomputed `merged_dataset.jsonl`, `cv_splits.json`, and all result files are included in `data/` so you can inspect paper numbers without re-running API calls.

---

## Running the Experiments

### 1. RetroBug (agentic pipeline)

Uses Claude Sonnet 4.5 via API by default. Expect ~798 commits × ~4 agent steps × API latency (~20–30 s/commit).

```bash
# Full 5-fold evaluation (~798 commits, resumable)
python3 run_experiment.py

# Single fold (pilot)
python3 run_experiment.py --fold 0

# Limit commits (quick smoke test)
python3 run_experiment.py --fold 0 --limit 5

# Dry run — no API calls; inspect KB + tools for 1 commit
python3 run_experiment.py --dry-run --fold 0 --limit 1

# Resume interrupted run (default behavior)
python3 run_experiment.py --resume

# Start fold from scratch
python3 run_experiment.py --no-resume --fold 0

# Recompute metrics only (no API)
python3 run_experiment.py --metrics-only
```

**Agent tools** (`tools.py`): `search_knowledge_base`, `analyze_diff_patterns`, `get_diff_summary`, `check_behavioral_change`, `analyze_message_intent`, `submit_answer`

### 2. Classical baselines (fast, no API)

```bash
# All 11 classical baselines, all 5 folds (~minutes)
python3 run_baselines.py --baseline all_classical

# Single baseline
python3 run_baselines.py --baseline tfidf_message_lr --fold 0

# Multiple baselines
python3 run_baselines.py --baseline keyword,knn_retrieval,hybrid_features_lr

# Pilot
python3 run_baselines.py --baseline keyword --fold 0 --limit 10
```

**Classical baseline names:**

| Name | Description |
|------|-------------|
| `keyword` | Fix-keyword heuristic on masked message |
| `tfidf_message_{lr,rf,xgb}` | TF-IDF on masked message only |
| `tfidf_message_diff_{lr,rf,xgb}` | TF-IDF on message + diff summary |
| `hybrid_features_{lr,rf,xgb}` | 9 hand-crafted features |
| `knn_retrieval` | Top-1 KB neighbor label (no LLM) |

### 3. LLM baselines (API cost / local GPU)

```bash
# Claude zero-shot — message only
python3 run_baselines.py --baseline claude_message --fold 0

# Claude zero-shot — message + diff
python3 run_baselines.py --baseline claude_message_diff --fold 0

# GPT-4o zero-shot (CCS-style prompt)
python3 run_baselines.py --baseline gpt4o_message_diff --fold 0

# CodeLlama-13b local zero-shot
python3 run_baselines.py --baseline codellama_message_diff --fold 0

# All LLM baselines
python3 run_baselines.py --baseline all_llm
```

Both `run_experiment.py` and `run_baselines.py` support `--resume` (default), `--no-resume`, `--fold N`, and `--limit N`.

### 4. Summarize and compare

```bash
# RetroBug cross-fold summary
python3 run_experiment.py --metrics-only
# → data/cross_fold_summary.txt

# Compare all baselines vs RetroBug
python3 summarize_baselines.py
# → data/baselines/baselines_comparison.txt
```

---

## Outputs and Results

### Per-fold outputs (RetroBug)

```
data/fold_{0..4}/
├── predictions.jsonl      # one JSON record per commit
├── metrics.json           # machine-readable metrics
├── metrics.txt            # human-readable report
├── knowledge_base.joblib  # fold-specific KB (built on first run)
└── token_usage.log        # API token summary (if applicable)
```

### Per-fold outputs (baselines)

```
data/baselines/<baseline_name>/fold_{0..4}/
├── predictions.jsonl
├── metrics.json
└── metrics.txt
```

### Key aggregate results (paper)

| Method | BF F1 | Acc. | MCC | Scope |
|--------|-------|------|-----|-------|
| **RetroBug** | **0.778** | **0.871** | **0.703** | 5-fold, n=798 |
| TF-IDF + LR (msg) | 0.500 | 0.732 | 0.319 | 5-fold, n=798 |
| kNN retrieval | 0.423 | 0.743 | 0.264 | 5-fold, n=798 |
| GPT-4o zero-shot | 0.629 | 0.794 | 0.493 | fold 0, n=160 |
| Claude zero-shot | 0.598 | 0.731 | 0.446 | fold 0, n=160 |
| CodeLlama-13b | 0.431 | 0.585 | 0.171 | fold 0, n=159* |

\*One CodeLlama response was unparseable and excluded from metrics.

### `predictions.jsonl` record fields

**RetroBug:** `commit_sha`, `human_label`, `predicted_label`, `confidence`, `reasoning`, `backward_reasoning`, `tool_calls`, `total_steps`, `latency_seconds`, `total_input_tokens`, `total_output_tokens`, `error`

**Baselines:** `commit_sha`, `human_label`, `predicted_label`, `baseline`, `confidence`, `reasoning`, `latency_seconds`, `error`

---

## Reproducing Paper Tables

Pre-generated LaTeX is in `paper_tables/`:

| File | Content |
|------|---------|
| `rq1_results.tex` | RQ1: RetroBug vs. classical baselines |
| `rq1_results_text.tex` | RQ1 results prose + summary box |
| `rq2_results.tex` | RQ2: RetroBug vs. zero-shot LLMs (fold 0) |
| `rq2_results_text.tex` | RQ2 results prose + summary box |
| `rq3_results.tex` | RQ3: cross-fold per-class stability |
| `rq3_results_text.tex` | RQ3 results prose + summary box |
| `confusion_matrix.tex` | Aggregate confusion matrix (n=798) |
| `baselines.tex` | Baseline descriptions |
| `evaluation_metrics.tex` | Metric definitions |
| `prompt_templates.tex` | Agent system + user prompts |

Regenerate per-fold comparison tables from cached metrics:

```bash
python3 paper_tables/generate_per_fold_tables.py
# reads  paper_tables/per_fold_metrics.json
# writes paper_tables/full_per_class_per_fold.{tex,md}
```

To regenerate `per_fold_metrics.json` from raw predictions, re-run all baselines and agent, then update the JSON manually or via a custom script.

---

## Verification Checklist

Use this checklist to confirm your environment reproduces the paper results:

- [ ] `python3 run_experiment.py --prepare-data` produces 798 commits in `merged_dataset.jsonl`
- [ ] `python3 run_experiment.py --metrics-only` reports RetroBug BF F1 ≈ **0.778 ± 0.027**
- [ ] `python3 summarize_baselines.py` shows TF-IDF + LR BF F1 ≈ **0.500**
- [ ] `data/fold_0/metrics.txt` shows RetroBug BF F1 ≈ **0.776**, MCC ≈ **0.698**
- [ ] `data/baselines/gpt4o_message_diff/fold_0/metrics.txt` shows BF F1 ≈ **0.629**
- [ ] Aggregate confusion matrix: TP=180, TN=515, FP=84, FN=19 (see `confusion_matrix.tex`)

---

## Troubleshooting

**`FileNotFoundError` for source CSVs**
Ensure `bugfix_pipeline/data/dataset{1,2}_100fix_300nonfix.csv` exist relative to the parent `BugFix/` directory.

**`ModuleNotFoundError: claude_message_experiment`**
The sibling directory `../claude_message_experiment/` must be present for Claude/GPT-4o/CodeLlama baselines.

**Anthropic / OpenAI authentication errors**
Check `.env` has valid `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. Scripts load `.env` automatically on startup.

**CodeLlama OOM**
CodeLlama-13b requires ~26 GB VRAM in float16. Set `CUDA_VISIBLE_DEVICES` to a free GPU and verify `CODELLAMA_PATH`.

**Interrupted runs**
Both pipelines default to `--resume`. Successfully classified commits are skipped; only pending or errored commits are retried. Use `--no-resume` to overwrite a fold.

**CodeLlama parse errors**
Some commits may return prose or code instead of JSON. The parser includes a prose fallback; unparseable responses are recorded with `predicted_label: null` and excluded from metrics.

---

## Citation

If you use this replication package, please cite our paper:

```bibtex
@inproceedings{retrobug2026,
  title   = {RetroBug: Agentic Backward Case-Based Reasoning for Bug-Fix Commit Classification},
  author  = {...},
  booktitle = {...},
  year    = {2026}
}
```

---

## License

See the parent repository for license terms. Do not commit `.env` files containing API keys.
