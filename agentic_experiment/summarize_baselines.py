#!/usr/bin/env python3
"""Summarize all baseline results and compare with the agentic system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.registry import BASELINE_NAMES
from compute_metrics import compute_fold_metrics
from config import DATA_DIR, N_FOLDS

BASELINES_DIR = DATA_DIR / "baselines"
REPORT_PATH = BASELINES_DIR / "baselines_comparison.txt"


def _load_predictions(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _collect_baseline(name: str) -> list[dict]:
    all_recs = []
    for fi in range(N_FOLDS):
        path = BASELINES_DIR / name / f"fold_{fi}" / "predictions.jsonl"
        if path.exists():
            all_recs.extend(_load_predictions(path))
    return all_recs


def _agent_predictions() -> list[dict]:
    all_recs = []
    for fi in range(N_FOLDS):
        path = DATA_DIR / f"fold_{fi}" / "predictions.jsonl"
        if path.exists():
            all_recs.extend(_load_predictions(path))
    return all_recs


def _fold_f1s(name: str, is_agent: bool = False) -> list[float]:
    f1s = []
    for fi in range(N_FOLDS):
        if is_agent:
            path = DATA_DIR / f"fold_{fi}" / "predictions.jsonl"
        else:
            path = BASELINES_DIR / name / f"fold_{fi}" / "predictions.jsonl"
        if not path.exists():
            continue
        m = compute_fold_metrics(_load_predictions(path))
        f1s.append(m["f1"])
    return f1s


def main():
    lines = [
        "=" * 90,
        "  BASELINE COMPARISON vs AGENTIC CBR (same 5-fold CV, n=798)",
        "=" * 90,
        "",
        f"{'Method':<28} {'F1 mean±std':<16} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'n':<6}",
        "-" * 90,
    ]

    # Agent first
    agent_recs = _agent_predictions()
    if agent_recs:
        fold_f1s = _fold_f1s("agent", is_agent=True)
        m = compute_fold_metrics(agent_recs)
        lines.append(
            f"{'Agentic CBR (Claude)':<28} "
            f"{np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}  "
            f"{m['accuracy']:.4f}   {m['precision']:.4f}   {m['recall']:.4f}   {m['n_total']}"
        )

    rows_for_table = []
    for name in BASELINE_NAMES:
        recs = _collect_baseline(name)
        if not recs:
            continue
        fold_f1s = _fold_f1s(name)
        m = compute_fold_metrics(recs)
        display = name.replace("_", " ")
        row = (
            f"{display:<28} "
            f"{np.mean(fold_f1s):.4f}±{np.std(fold_f1s):.4f}  "
            f"{m['accuracy']:.4f}   {m['precision']:.4f}   {m['recall']:.4f}   {m['n_total']}"
        )
        lines.append(row)
        rows_for_table.append((name, m, recs))

    lines.extend(["", "=" * 90, "  PER-CLASS REPORTS (aggregate over all folds)", "=" * 90])

    if agent_recs:
        yt = [r["human_label"] for r in agent_recs]
        yp = [r["predicted_label"] for r in agent_recs]
        lines.extend([
            "",
            "--- Agentic CBR (Claude) ---",
            classification_report(yt, yp, target_names=["Non-Bug-Fix", "Bug-Fix"], digits=4),
        ])

    for name, m, recs in rows_for_table:
        yt = [r["human_label"] for r in recs]
        yp = [r["predicted_label"] for r in recs if r.get("predicted_label") in (0, 1)]
        yt_valid = [r["human_label"] for r in recs if r.get("predicted_label") in (0, 1)]
        lines.extend([
            "",
            f"--- {name} ---",
            f"Confusion: TP={m['confusion']['tp']} FN={m['confusion']['fn']} "
            f"FP={m['confusion']['fp']} TN={m['confusion']['tn']}",
            classification_report(yt_valid, yp, target_names=["Non-Bug-Fix", "Bug-Fix"], digits=4),
        ])

    report = "\n".join(lines)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
