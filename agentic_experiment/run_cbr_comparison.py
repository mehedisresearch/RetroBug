#!/usr/bin/env python3
"""
Run Forward vs Backward CBR comparison (simple 1-neighbor RAG, single LLM turn).

Same pipeline for both conditions:
  - fold-specific KB (all-MiniLM-L6-v2, top-1 neighbor)
  - masked message + diff preview for target
  - Claude Sonnet, temperature=0

Only difference: the classification prompt
  backward — WHY was the reference labeled? then classify target
  forward  — given this reference, classify the target

Usage:
    python run_cbr_comparison.py --fold 0
    python run_cbr_comparison.py --fold 0 --mode backward --limit 5
    python run_cbr_comparison.py --fold 0 --no-resume
    python summarize_cbr_comparison.py --fold 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cbr_rag.classifier import CBRRagClassifier
from compute_metrics import compute_fold_metrics, format_fold_report
from config import DATA_DIR, N_FOLDS
from knowledge_base import KnowledgeBase
from load_data import get_fold_data, load_splits

CBR_DIR = DATA_DIR / "cbr_rag"
MODES = ("backward", "forward")


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


def _mode_dir(mode: str, fold_idx: int) -> Path:
    return CBR_DIR / mode / f"fold_{fold_idx}"


def _predictions_path(mode: str, fold_idx: int) -> Path:
    return _mode_dir(mode, fold_idx) / "predictions.jsonl"


def _kb_path(fold_idx: int) -> Path:
    return DATA_DIR / f"fold_{fold_idx}" / "knowledge_base.joblib"


def _load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("predicted_label") in (0, 1):
                    done.add(rec["commit_sha"])
    return done


def _load_kb(train_rows: list[dict], fold_idx: int, *, resume: bool) -> KnowledgeBase:
    kb_path = _kb_path(fold_idx)
    kb = KnowledgeBase()
    if kb_path.exists() and resume:
        print(f"  Loading KB from {kb_path}")
        kb.load(kb_path)
    else:
        print(f"  Building KB from {len(train_rows)} train commits...")
        kb.build(train_rows)
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb.save(kb_path)
    print(f"  KB size: {kb.size} (top-1 neighbor per commit)")
    return kb


def run_mode_fold(
    mode: str,
    fold_idx: int,
    rows: list[dict],
    folds: list[dict],
    kb: KnowledgeBase,
    *,
    resume: bool = True,
    limit: int = 0,
) -> list[dict]:
    train_rows, test_rows = get_fold_data(rows, folds, fold_idx)
    del train_rows  # KB already built

    out_dir = _mode_dir(mode, fold_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = _predictions_path(mode, fold_idx)

    classifier = CBRRagClassifier(mode)
    print(f"\n  Mode: {mode} — {classifier.description}")
    classifier.fit([], kb=kb)
    classifier._get_client()

    done = _load_done(pred_path) if resume else set()
    pending = [r for r in test_rows if r["commit_sha"] not in done]
    if limit:
        pending = pending[:limit]

    print(f"  Fold {fold_idx}: {len(test_rows)} test, {len(done)} done, {len(pending)} pending")
    if not pending:
        print("  Nothing to do.")
        return []

    file_mode = "a" if resume and pred_path.exists() else "w"
    results: list[dict] = []

    with open(pred_path, file_mode, encoding="utf-8") as fout:
        for i, commit in enumerate(pending, 1):
            t0 = time.time()
            try:
                pred = classifier.predict(commit, kb=kb)
                rec = {
                    "commit_sha": commit["commit_sha"],
                    "repo_name": commit["repo_name"],
                    "human_label": commit["human_label"],
                    "masked_commit_message": commit["masked_commit_message"],
                    "cbr_mode": mode,
                    "predicted_label": pred.get("predicted_label"),
                    "confidence": pred.get("confidence"),
                    "reasoning": pred.get("reasoning", ""),
                    "backward_reasoning": pred.get("backward_reasoning", ""),
                    "kb_reference_sha": pred.get("kb_reference_sha"),
                    "kb_similarity": pred.get("kb_similarity"),
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
                    f"(ref={str(rec['kb_reference_sha'])[:10]}, "
                    f"sim={rec['kb_similarity']}, {rec['latency_seconds']:.1f}s)"
                )
            except Exception as exc:
                rec = {
                    "commit_sha": commit["commit_sha"],
                    "repo_name": commit["repo_name"],
                    "human_label": commit["human_label"],
                    "masked_commit_message": commit["masked_commit_message"],
                    "cbr_mode": mode,
                    "predicted_label": None,
                    "error": str(exc),
                }
                print(f"  [{i:>3}/{len(pending)}] {commit['commit_sha'][:10]} → ERROR: {exc}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(rec)

    all_preds = [
        json.loads(line)
        for line in pred_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = compute_fold_metrics(all_preds)
    report = format_fold_report(metrics, fold_idx).replace(
        "Agentic Classification",
        f"CBR RAG ({mode})",
    )
    (_mode_dir(mode, fold_idx) / "metrics.txt").write_text(report, encoding="utf-8")
    with open(_mode_dir(mode, fold_idx) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return results


def main():
    _load_env()
    parser = argparse.ArgumentParser(
        description="Forward vs Backward CBR (simple 1-neighbor RAG)"
    )
    parser.add_argument("--fold", type=int, default=0, help="CV fold (default: 0)")
    parser.add_argument(
        "--mode",
        choices=["both", *MODES],
        default="both",
        help="Run backward, forward, or both (default: both)",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max test commits")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    modes = list(MODES) if args.mode == "both" else [args.mode]
    rows, folds = load_splits()
    train_rows, _ = get_fold_data(rows, folds, args.fold)

    print(f"\nCBR RAG comparison — fold {args.fold}")
    print("Pipeline: top-1 KB neighbor → single Claude call (no tools, no loop)")

    kb = _load_kb(train_rows, args.fold, resume=args.resume)

    for mode in modes:
        print(f"\n{'=' * 60}\n  {mode.upper()} CBR\n{'=' * 60}")
        run_mode_fold(
            mode, args.fold, rows, folds, kb,
            resume=args.resume,
            limit=args.limit,
        )

    print(f"\nDone. Compare: python summarize_cbr_comparison.py --fold {args.fold}")


if __name__ == "__main__":
    main()
