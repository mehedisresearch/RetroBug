"""
Compute classification metrics for the agentic experiment.
Supports per-fold metrics and cross-fold aggregation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return round((tp * tn - fp * fn) / denom, 4) if denom else 0.0


def compute_fold_metrics(predictions: list[dict]) -> dict[str, Any]:
    """Compute metrics for one fold's predictions."""
    valid = [p for p in predictions if p.get("predicted_label") in (0, 1)]
    errors = [p for p in predictions if p.get("predicted_label") not in (0, 1)]

    tp = fp = tn = fn = 0
    for p in valid:
        pred = p["predicted_label"] == 1
        truth = p["human_label"] == 1
        if pred and truth:
            tp += 1
        elif pred:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1

    n_valid = len(valid)
    n_total = len(predictions)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    avg_confidence = 0.0
    if valid:
        avg_confidence = round(sum(p.get("confidence") or 0 for p in valid) / len(valid), 1)

    avg_steps = 0.0
    avg_tools = 0.0
    avg_latency = 0.0
    avg_tokens = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    if predictions:
        avg_steps = round(sum(p.get("total_steps", 0) for p in predictions) / len(predictions), 2)
        avg_tools = round(sum(len(p.get("tool_calls", [])) for p in predictions) / len(predictions), 2)
        avg_latency = round(sum(p.get("latency_seconds", 0) for p in predictions) / len(predictions), 2)
        total_input_tokens = sum(p.get("total_input_tokens", 0) for p in predictions)
        total_output_tokens = sum(p.get("total_output_tokens", 0) for p in predictions)
        avg_tokens = round(
            (total_input_tokens + total_output_tokens) / len(predictions), 0
        )

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_errors": len(errors),
        "coverage": _safe_div(n_valid, n_total),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": _mcc(tp, tn, fp, fn),
        "accuracy": _safe_div(tp + tn, n_valid),
        "avg_confidence": avg_confidence,
        "avg_steps": avg_steps,
        "avg_tool_calls": avg_tools,
        "avg_latency_s": avg_latency,
        "avg_tokens": avg_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


def compute_cross_fold_metrics(fold_metrics: list[dict]) -> dict[str, Any]:
    """Aggregate metrics across all folds (mean + std)."""
    import numpy as np

    keys = ["precision", "recall", "f1", "mcc", "accuracy",
            "avg_confidence", "avg_steps", "avg_tool_calls",
            "avg_latency_s", "avg_tokens"]

    summary = {"n_folds": len(fold_metrics)}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }

    total_tp = sum(m["confusion"]["tp"] for m in fold_metrics)
    total_fp = sum(m["confusion"]["fp"] for m in fold_metrics)
    total_tn = sum(m["confusion"]["tn"] for m in fold_metrics)
    total_fn = sum(m["confusion"]["fn"] for m in fold_metrics)

    summary["aggregate_confusion"] = {
        "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn,
    }
    summary["aggregate_precision"] = _safe_div(total_tp, total_tp + total_fp)
    summary["aggregate_recall"] = _safe_div(total_tp, total_tp + total_fn)
    agg_p = summary["aggregate_precision"]
    agg_r = summary["aggregate_recall"]
    summary["aggregate_f1"] = _safe_div(2 * agg_p * agg_r, agg_p + agg_r)
    summary["aggregate_mcc"] = _mcc(total_tp, total_tn, total_fp, total_fn)
    summary["aggregate_accuracy"] = _safe_div(total_tp + total_tn, total_tp + total_fp + total_tn + total_fn)

    return summary


def format_fold_report(metrics: dict, fold_idx: int) -> str:
    """Format a human-readable report for one fold."""
    cm = metrics["confusion"]
    lines = [
        "=" * 60,
        f"  Agentic Classification — Fold {fold_idx}  (n={metrics['n_total']})",
        "=" * 60,
        "",
        f"  Coverage   : {metrics['coverage'] * 100:.1f}%  "
        f"({metrics['n_valid']} valid / {metrics['n_errors']} errors)",
        "",
        "  Confusion Matrix (rows=actual, cols=predicted):",
        "                    Pred Bug-Fix  Pred Non-Bug",
        f"  Actual Bug-Fix      {cm['tp']:>6}        {cm['fn']:>6}",
        f"  Actual Non-Bug      {cm['fp']:>6}        {cm['tn']:>6}",
        "",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1        : {metrics['f1']:.4f}",
        f"  MCC       : {metrics['mcc']:.4f}",
        f"  Accuracy  : {metrics['accuracy']:.4f}",
        "",
        "  Agent Efficiency:",
        f"    Avg Steps/Commit   : {metrics['avg_steps']:.1f}",
        f"    Avg Tool Calls     : {metrics['avg_tool_calls']:.1f}",
        f"    Avg Latency        : {metrics['avg_latency_s']:.1f}s",
        f"    Avg Tokens/Commit  : {metrics['avg_tokens']:.0f}",
        f"    Total Input Tokens : {metrics.get('total_input_tokens', 0):,}",
        f"    Total Output Tokens: {metrics.get('total_output_tokens', 0):,}",
        f"    Total Tokens       : {metrics.get('total_tokens', 0):,}",
        f"    Avg Confidence     : {metrics['avg_confidence']:.1f}",
        "",
    ]
    return "\n".join(lines)


def format_summary_report(summary: dict) -> str:
    """Format the cross-fold summary."""
    lines = [
        "=" * 60,
        f"  CROSS-FOLD SUMMARY ({summary['n_folds']} folds)",
        "=" * 60,
        "",
    ]

    for key in ["precision", "recall", "f1", "mcc", "accuracy"]:
        if key in summary:
            v = summary[key]
            lines.append(f"  {key:<12}: {v['mean']:.4f} ± {v['std']:.4f}  "
                        f"(min={v['min']:.4f}, max={v['max']:.4f})")

    lines.append("")
    lines.append("  Agent Efficiency (mean ± std):")
    for key in ["avg_steps", "avg_tool_calls", "avg_latency_s", "avg_tokens"]:
        if key in summary:
            v = summary[key]
            lines.append(f"    {key:<20}: {v['mean']:.2f} ± {v['std']:.2f}")

    lines.append("")
    cm = summary["aggregate_confusion"]
    lines.append("  Aggregate Confusion Matrix:")
    lines.append("                    Pred Bug-Fix  Pred Non-Bug")
    lines.append(f"  Actual Bug-Fix      {cm['tp']:>6}        {cm['fn']:>6}")
    lines.append(f"  Actual Non-Bug      {cm['fp']:>6}        {cm['tn']:>6}")
    lines.append("")
    lines.append(f"  Aggregate F1  : {summary['aggregate_f1']:.4f}")
    lines.append(f"  Aggregate MCC : {summary['aggregate_mcc']:.4f}")
    lines.append(f"  Aggregate Acc : {summary['aggregate_accuracy']:.4f}")
    lines.append("")

    return "\n".join(lines)


def write_fold_metrics(predictions_path: Path, fold_idx: int) -> dict:
    """Load predictions JSONL and write metrics for one fold."""
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    metrics = compute_fold_metrics(predictions)
    report = format_fold_report(metrics, fold_idx)

    metrics_json_path = predictions_path.parent / "metrics.json"
    metrics_txt_path = predictions_path.parent / "metrics.txt"

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics_txt_path.write_text(report, encoding="utf-8")

    print(report)
    return metrics
