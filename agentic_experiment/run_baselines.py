#!/usr/bin/env python3
"""
Run baseline classifiers on the same 5-fold CV splits as the agentic system.

Usage:
    # All fast classical baselines, all folds
    python run_baselines.py --baseline all_classical

    # Single baseline, single fold
    python run_baselines.py --baseline knn_retrieval --fold 0

    # LLM baselines (API cost)
    python run_baselines.py --baseline claude_message_diff --fold 0
    python run_baselines.py --baseline gpt4o_message_diff --fold 0

    # Summarize after running
    python summarize_baselines.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_metrics import compute_fold_metrics, format_fold_report
from config import DATA_DIR, N_FOLDS
from knowledge_base import KnowledgeBase
from load_data import get_fold_data, load_splits
from baselines.registry import BASELINE_NAMES, CLASSICAL_ONLY, LLM_ONLY, get_baseline

BASELINES_DIR = DATA_DIR / "baselines"


def _load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _baseline_dir(name: str, fold_idx: int) -> Path:
    return BASELINES_DIR / name / f"fold_{fold_idx}"


def _predictions_path(name: str, fold_idx: int) -> Path:
    return _baseline_dir(name, fold_idx) / "predictions.jsonl"


def _kb_path(fold_idx: int) -> Path:
    agent_kb = DATA_DIR / f"fold_{fold_idx}" / "knowledge_base.joblib"
    if agent_kb.exists():
        return agent_kb
    return _baseline_dir("knn_retrieval", fold_idx) / "knowledge_base.joblib"


def _load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("predicted_label") in (0, 1):
                        done.add(rec["commit_sha"])
    return done


def run_baseline_fold(
    baseline_name: str,
    fold_idx: int,
    rows: list[dict],
    folds: list[dict],
    *,
    resume: bool = True,
    limit: int = 0,
) -> list[dict]:
    train_rows, test_rows = get_fold_data(rows, folds, fold_idx)
    out_dir = _baseline_dir(baseline_name, fold_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = _predictions_path(baseline_name, fold_idx)

    classifier = get_baseline(baseline_name)
    print(f"\n  Baseline: {baseline_name} — {classifier.description}")

    if baseline_name == "gpt4o_message_diff":
        classifier._get_client()
    elif baseline_name == "codellama_message_diff":
        import local_llm
        local_llm.set_backend("codellama")
        local_llm._load_model()
    elif baseline_name.startswith("claude_"):
        classifier._get_client()

    kb = None
    if baseline_name == "knn_retrieval":
        kb_path = _kb_path(fold_idx)
        kb = KnowledgeBase()
        if kb_path.exists() and resume:
            kb.load(kb_path)
        else:
            print(f"  Building KB from {len(train_rows)} train commits...")
            kb.build(train_rows)
            kb.save(kb_path)
        print(f"  KB size: {kb.size}")
    else:
        print(f"  Training on {len(train_rows)} commits...")
        classifier.fit(train_rows)

    done = _load_done(pred_path) if resume else set()
    pending = [r for r in test_rows if r["commit_sha"] not in done]
    if limit:
        pending = pending[:limit]

    print(f"  Fold {fold_idx}: {len(test_rows)} test, {len(done)} done, {len(pending)} pending")
    if not pending:
        print("  Nothing to do.")
        return []

    mode = "a" if resume and pred_path.exists() else "w"
    results = []

    with open(pred_path, mode, encoding="utf-8") as fout:
        for i, commit in enumerate(pending, 1):
            t0 = time.time()
            try:
                pred = classifier.predict(commit, kb=kb)
                rec = {
                    "commit_sha": commit["commit_sha"],
                    "repo_name": commit["repo_name"],
                    "human_label": commit["human_label"],
                    "masked_commit_message": commit["masked_commit_message"],
                    "baseline": baseline_name,
                    "predicted_label": pred.get("predicted_label"),
                    "confidence": pred.get("confidence"),
                    "reasoning": pred.get("reasoning", ""),
                    "error": pred.get("error"),
                    "latency_seconds": pred.get("latency_seconds", round(time.time() - t0, 2)),
                    "total_input_tokens": pred.get("total_input_tokens", 0),
                    "total_output_tokens": pred.get("total_output_tokens", 0),
                }
                label_str = (
                    "BUG" if rec["predicted_label"] == 1
                    else "NON" if rec["predicted_label"] == 0
                    else "ERR"
                )
                print(
                    f"  [{i:>3}/{len(pending)}] {commit['commit_sha'][:10]} → {label_str} "
                    f"({rec['latency_seconds']:.1f}s)"
                )
            except Exception as exc:
                rec = {
                    "commit_sha": commit["commit_sha"],
                    "repo_name": commit["repo_name"],
                    "human_label": commit["human_label"],
                    "masked_commit_message": commit["masked_commit_message"],
                    "baseline": baseline_name,
                    "predicted_label": None,
                    "error": str(exc),
                }
                print(f"  [{i:>3}/{len(pending)}] {commit['commit_sha'][:10]} → ERROR: {exc}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(rec)

    metrics = compute_fold_metrics(
        [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    )
    report = format_fold_report(metrics, fold_idx).replace(
        "Agentic Classification", f"Baseline: {baseline_name}"
    )
    (_baseline_dir(baseline_name, fold_idx) / "metrics.txt").write_text(report, encoding="utf-8")
    with open(_baseline_dir(baseline_name, fold_idx) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return results


def main():
    _load_env()
    parser = argparse.ArgumentParser(description="Run baseline classifiers")
    parser.add_argument(
        "--baseline",
        required=True,
        help=(
            f"Baseline name, 'all_classical', 'all_llm', or 'all'. "
            f"Choices: {', '.join(BASELINE_NAMES)}"
        ),
    )
    parser.add_argument("--fold", type=int, default=None, help="Single fold (0-4)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    if args.baseline == "all_classical":
        names = CLASSICAL_ONLY
    elif args.baseline == "all_llm":
        names = LLM_ONLY
    elif args.baseline == "all":
        names = BASELINE_NAMES
    elif "," in args.baseline:
        names = [n.strip() for n in args.baseline.split(",") if n.strip()]
    else:
        names = [args.baseline]

    rows, folds = load_splits()
    fold_indices = [args.fold] if args.fold is not None else list(range(N_FOLDS))

    print(f"Dataset: {len(rows)} commits | Baselines: {names}")

    for bname in names:
        print(f"\n{'='*60}\n  BASELINE: {bname}\n{'='*60}")
        for fi in fold_indices:
            run_baseline_fold(
                bname, fi, rows, folds,
                resume=args.resume,
                limit=args.limit,
            )

    print(f"\nDone. Run: python summarize_baselines.py")


if __name__ == "__main__":
    main()
