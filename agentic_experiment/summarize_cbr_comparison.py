#!/usr/bin/env python3
"""Summarize CBR mechanism comparison: forward/backward RAG + agent variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_metrics import compute_fold_metrics
from config import DATA_DIR

CBR_DIR = DATA_DIR / "cbr_rag"
ABLATION_DIR = DATA_DIR / "ablation"

# Ordered mechanism ladder (fold-level predictions)
CONDITIONS: list[tuple[str, str]] = [
    ("Forward CBR", "cbr_rag:forward"),
    ("Backward CBR", "cbr_rag:backward"),
    ("RetroBug (no tools)", "ablation:no_tools"),
    ("RetroBug (full)", "main"),
]


def _resolve_path(source: str, fold_idx: int) -> Path | None:
    if source.startswith("cbr_rag:"):
        mode = source.split(":", 1)[1]
        path = CBR_DIR / mode / f"fold_{fold_idx}" / "predictions.jsonl"
    elif source.startswith("ablation:"):
        variant = source.split(":", 1)[1]
        path = ABLATION_DIR / variant / f"fold_{fold_idx}" / "predictions.jsonl"
    elif source == "main":
        path = DATA_DIR / f"fold_{fold_idx}" / "predictions.jsonl"
    else:
        return None
    return path if path.exists() else None


def _load_predictions(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _per_class_rows(name: str, preds: list[dict]) -> tuple[dict, dict | None]:
    """Return aggregate metrics + sklearn per-class dict (if sklearn available)."""
    m = compute_fold_metrics(preds)
    per_class = None
    try:
        from sklearn.metrics import classification_report

        valid = [p for p in preds if p.get("predicted_label") in (0, 1)]
        yt = [p["human_label"] for p in valid]
        yp = [p["predicted_label"] for p in valid]
        per_class = classification_report(
            yt, yp,
            target_names=["Non-Bug-Fix", "Bug-Fix"],
            output_dict=True,
            zero_division=0,
        )
    except ImportError:
        pass
    return m, per_class


def summarize(fold_idx: int) -> list[str]:
    lines = [
        "=" * 78,
        f"  CBR MECHANISM LADDER — Fold {fold_idx}",
        "=" * 78,
        "",
        "Simple RAG: top-1 neighbor, single Claude turn, no tools, no loop.",
        "No tools: backward CBR + agent loop, analysis tools disabled.",
        "Full: backward CBR + agent loop + all tools.",
        "",
        f"{'Condition':<22} {'BF F1':>8} {'NB F1':>8} {'Acc':>8} {'MCC':>8} {'Cov':>6}",
        "-" * 78,
    ]

    loaded: list[tuple[str, dict, dict | None]] = []
    for name, source in CONDITIONS:
        path = _resolve_path(source, fold_idx)
        if path is None:
            lines.append(f"{name:<22}  (missing — {source})")
            continue
        preds = _load_predictions(path)
        m, per_class = _per_class_rows(name, preds)
        loaded.append((name, m, per_class))
        nb_f1 = per_class["Non-Bug-Fix"]["f1-score"] if per_class else float("nan")
        cov = f"{m['coverage'] * 100:.0f}%"
        lines.append(
            f"{name:<22} {m['f1']:>8.4f} {nb_f1:>8.4f} {m['accuracy']:>8.4f} "
            f"{m['mcc']:>8.4f} {cov:>6}"
        )

    # Per-class detail table
    if any(pc for _, _, pc in loaded):
        lines.extend([
            "",
            "Per-class detail:",
            f"{'Condition':<22} {'Class':<14} {'Prec':>7} {'Rec':>7} {'F1':>7}",
            "-" * 78,
        ])
        for name, _, per_class in loaded:
            if not per_class:
                continue
            for i, cls in enumerate(["Non-Bug-Fix", "Bug-Fix"]):
                d = per_class[cls]
                lines.append(
                    f"{name if i == 0 else '':<22} {cls:<14} "
                    f"{d['precision']:>7.4f} {d['recall']:>7.4f} {d['f1-score']:>7.4f}"
                )

    # Deltas vs forward RAG
    by_name = {n: m for n, m, _ in loaded}
    if "Forward CBR" in by_name and "Backward CBR" in by_name:
        lines.extend([
            "",
            "Delta (Backward CBR − Forward CBR):",
            f"  BF F1: {by_name['Backward CBR']['f1'] - by_name['Forward CBR']['f1']:+.4f}",
            f"  MCC:   {by_name['Backward CBR']['mcc'] - by_name['Forward CBR']['mcc']:+.4f}",
        ])
    if "Forward CBR" in by_name and "RetroBug (full)" in by_name:
        lines.extend([
            "",
            "Delta (Full − Forward CBR):",
            f"  BF F1: {by_name['RetroBug (full)']['f1'] - by_name['Forward CBR']['f1']:+.4f}",
            f"  MCC:   {by_name['RetroBug (full)']['mcc'] - by_name['Forward CBR']['mcc']:+.4f}",
        ])
    if "RetroBug (no tools)" in by_name and "RetroBug (full)" in by_name:
        lines.extend([
            "",
            "Delta (Full − No tools):",
            f"  BF F1: {by_name['RetroBug (full)']['f1'] - by_name['RetroBug (no tools)']['f1']:+.4f}",
            f"  MCC:   {by_name['RetroBug (full)']['mcc'] - by_name['RetroBug (no tools)']['mcc']:+.4f}",
        ])

    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    lines = summarize(args.fold)
    text = "\n".join(lines)
    print(text)

    CBR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CBR_DIR / f"comparison_fold_{args.fold}.txt"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
