# RetroBug Replication Package — Git Upload Guide

Replication scope: **RQ1, RQ2, RQ3 only** (no ablation, no model-comparison extras).

## What to push

```
BugFix/
├── REPLICATION.md
├── scripts/package_replication.sh
├── agentic_experiment/              ← MAIN PACKAGE
│   ├── README.md
│   ├── requirements.txt, .env.example, .gitignore
│   ├── run_experiment.py              ← RetroBug (RQ1–RQ3)
│   ├── run_baselines.py               ← classical + LLM baselines
│   ├── summarize_baselines.py
│   ├── agent.py, tools.py, knowledge_base.py, …
│   ├── baselines/
│   ├── data/
│   │   ├── merged_dataset.jsonl, cv_splits.json
│   │   ├── fold_{0..4}/             ← RetroBug predictions + metrics
│   │   ├── cross_fold_summary.*       ← RQ3
│   │   └── baselines/
│   │       ├── keyword, tfidf_message_*, hybrid_features_*, knn_retrieval/  (RQ1, 5-fold)
│   │       ├── claude_message_diff, gpt4o_message_diff, codellama_message_diff/  (RQ2, fold 0)
│   │       └── baselines_comparison.txt
│   └── paper_tables/                  ← RQ1–RQ3 tables + baselines description
├── bugfix_pipeline/data/*.csv
└── claude_message_experiment/{prompts,parse_response}.py
```

## What is excluded (kept locally only)

| Path | Reason |
|------|--------|
| `data/ablation/`, `ablation/`, `run_ablation.py` | Ablation study |
| `data/model_comparison/`, `run_model_comparison.py` | Extra experiment |
| `data/baselines/claude_message/` | Not in RQ2 table (use `claude_message_diff`) |
| `data/baselines/tfidf_message_diff_*` | Not in RQ1 paper table |
| `data/*_run.log`, `token_usage.log` | Run logs |
| `*.joblib` | KB caches (rebuilt on first run) |
| `paper_tables/discussion.tex`, etc. | Paper prose, not needed to verify numbers |
| `.env` | API keys |

## Push to GitHub

```bash
cd ~/BugFix
git branch -M main
git push -u origin main
```

If push is blocked by **secret scanning**, AWS keys in historical commit diffs must be
redacted (`AKIA...` → `AKIA_REDACTED_KEY_ID`) in `merged_dataset.jsonl` and CSVs.

## Verify results (no API calls)

```bash
cd agentic_experiment
python3 run_experiment.py --metrics-only      # RQ1 + RQ3: BF F1 ≈ 0.778
python3 summarize_baselines.py                # RQ1 + RQ2 comparison table
```

## Entry point

**`agentic_experiment/README.md`**
