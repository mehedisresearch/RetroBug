# RetroBug Replication Package

Replication artifact for **RetroBug** — an agentic backward case-based reasoning (CBR) system for binary bug-fix commit classification (Bug-Fix vs. Non-Bug-Fix).

This package contains experiment code, evaluation scripts, and precomputed results for **RQ1–RQ3**. All methods use the same **5-fold stratified cross-validation** protocol on **798 commits** (seed=42).

---

## Research Questions

**RQ1: How effective is RetroBug compared with non-agentic baselines for bug-fix commit classification?**

We evaluate whether RetroBug improves bug-fix commit classification over heuristic, classical machine learning, and retrieval-based methods (keyword, TF-IDF, hybrid features, kNN retrieval). Evaluation uses full 5-fold CV (n=798).

**RQ2: Does the RetroBug agentic pipeline improve classification over Forward Case-Based Reasoning (Forward CBR)?**

We compare RetroBug with a **Forward CBR** baseline using the **same retrieval setting** (top-1 KB neighbor) and the **same language model** (Claude Sonnet), but with a single-turn prompt: *“Given this reference, classify the target.”* RetroBug adds backward CBR, analysis tools, and a multi-step agent loop. RQ2 is evaluated on **fold 0** (n=160); re-run or extend with `run_cbr_comparison.py`.

**RQ3: How stable is RetroBug across cross-validation folds?**

We evaluate whether RetroBug performs consistently across data splits and maintains balanced performance on both Bug-Fix and Non-Bug-Fix commits (5-fold per-class metrics).

> **Discussion (not an RQ):** Comparison with zero-shot LLM baselines (Claude, GPT-4o, CodeLlama) is reported in the Discussion section. Precomputed LLM baseline outputs for fold 0 remain in `data/baselines/` for verification.

> **Note:** Ablation studies (`data/ablation/`) are maintained locally and are not part of this replication package.

---

## Repository Layout

```
RetroBug/
├── README.md
├── scripts/package_replication.sh
├── agentic_experiment/
│   ├── run_experiment.py           ← RetroBug (RQ1, RQ3)
│   ├── run_baselines.py            ← non-agentic baselines (RQ1)
│   ├── run_cbr_comparison.py       ← Forward / Backward CBR (RQ2)
│   ├── summarize_cbr_comparison.py
│   ├── cbr_rag/                    ← simple 1-neighbor RAG classifiers
│   └── data/
│       ├── fold_{0..4}/            ← RetroBug per-fold results
│       ├── baselines/              ← RQ1 classical + discussion LLM baselines
│       ├── cbr_rag/forward/        ← RQ2 Forward CBR (fold 0 bundled)
│       └── cross_fold_summary.txt  ← RQ3
├── bugfix_pipeline/data/           ← source CSV datasets
└── claude_message_experiment/      ← shared LLM prompt/parser
```

---

## Installation

```bash
cd agentic_experiment
python3 -m pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY
```

---

## Running the Experiments

### RQ1 — RetroBug and non-agentic baselines

```bash
cd agentic_experiment
python3 run_experiment.py --metrics-only          # RetroBug 5-fold summary
python3 run_baselines.py --baseline all_classical # classical + kNN
python3 summarize_baselines.py
```

### RQ2 — Forward CBR vs RetroBug

```bash
# Forward CBR on fold 0 (single-turn, top-1 neighbor, no tools)
python3 run_cbr_comparison.py --fold 0 --mode forward

# Compare with bundled / local results
python3 summarize_cbr_comparison.py --fold 0
```

### RQ3 — Cross-fold stability

```bash
python3 run_experiment.py --metrics-only   # → data/cross_fold_summary.txt
```

---

## Key Results

### RQ1 — RetroBug vs non-agentic baselines (5-fold, n=798)

| Method | BF F1 | Acc. | MCC |
|--------|-------|------|-----|
| **RetroBug** | **0.778** | **0.871** | **0.703** |
| TF-IDF + LR (message) | 0.500 | 0.732 | 0.319 |
| kNN retrieval | 0.423 | 0.743 | 0.264 |

### RQ2 — RetroBug vs Forward CBR (fold 0, n=160)

| Method | NB F1 | BF F1 | Acc. | MCC |
|--------|-------|-------|------|-----|
| Forward CBR | 0.881 | 0.623 | 0.819 | 0.505 |
| **RetroBug (full)** | **0.919** | **0.777** | **0.881** | **0.698** |

Same KB (top-1), same model; Forward CBR = one LLM call, no tools, no agent loop.

### RQ3 — Cross-fold stability (RetroBug)

Bug-Fix F1 = **0.778 ± 0.027**; Non-Bug-Fix F1 = **0.909 ± 0.015**; MCC = **0.705 ± 0.035** across 5 folds. See `data/cross_fold_summary.txt`.

---

## Verification

```bash
cd agentic_experiment
python3 run_experiment.py --metrics-only
python3 summarize_baselines.py
python3 summarize_cbr_comparison.py --fold 0
```

- [ ] RetroBug pooled BF F1 ≈ **0.778** (`data/cross_fold_summary.txt`)
- [ ] Forward CBR fold 0 BF F1 ≈ **0.623** (`data/cbr_rag/forward/fold_0/metrics.txt`)
- [ ] RetroBug fold 0 BF F1 ≈ **0.777** (`data/fold_0/metrics.txt`)

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
