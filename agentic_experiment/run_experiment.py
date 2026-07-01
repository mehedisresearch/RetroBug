#!/usr/bin/env python3
"""
Agentic Bug-Fix Commit Classification Experiment.

Combines retrieval-augmented case-based reasoning with autonomous
tool-using agents for commit classification.

Usage:
    # Run full experiment (all folds)
    python run_experiment.py

    # Run a specific fold
    python run_experiment.py --fold 0

    # Resume from where it left off
    python run_experiment.py --resume

    # Recompute metrics only (no API calls)
    python run_experiment.py --metrics-only

    # Dry run (show prompts, no API calls)
    python run_experiment.py --dry-run --fold 0 --limit 1

    # Prepare data only (merge datasets, create splits)
    python run_experiment.py --prepare-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, MODEL_ID, N_FOLDS
from load_data import get_fold_data, load_splits, save_merged_and_splits
from knowledge_base import KnowledgeBase
from agent import run_agent, AgentTrace
from compute_metrics import (
    compute_cross_fold_metrics,
    compute_fold_metrics,
    format_fold_report,
    format_summary_report,
    write_fold_metrics,
)


def _load_env():
    """Load .env file if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and not os.environ.get(key, "").strip():
                os.environ[key] = val


def _load_model():
    """Pre-load the local model (called once before processing)."""
    from local_llm import _load_model
    _load_model()


def _fold_dir(fold_idx: int) -> Path:
    return DATA_DIR / f"fold_{fold_idx}"


def _predictions_path(fold_idx: int) -> Path:
    return _fold_dir(fold_idx) / "predictions.jsonl"


def _kb_path(fold_idx: int) -> Path:
    return _fold_dir(fold_idx) / "knowledge_base.joblib"


def _load_done_shas(fold_idx: int, include_errors: bool = True) -> set[str]:
    """Load already-completed commit SHAs for resume support."""
    path = _predictions_path(fold_idx)
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if include_errors or rec.get("predicted_label") is not None:
                            done.add(rec["commit_sha"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run_fold(
    fold_idx: int,
    rows: list[dict],
    folds: list[dict],
    *,
    resume: bool = True,
    limit: int = 0,
    dry_run: bool = False,
) -> list[dict]:
    """Run the agentic experiment on one fold."""
    train_rows, test_rows = get_fold_data(rows, folds, fold_idx)

    fold_dir = _fold_dir(fold_idx)
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_path = _predictions_path(fold_idx)
    kb_path = _kb_path(fold_idx)

    # Build or load KB
    kb = KnowledgeBase()
    if kb_path.exists() and resume:
        print(f"  Loading KB from {kb_path}")
        kb.load(kb_path)
    else:
        print(f"  Building KB from {len(train_rows)} training commits...")
        kb.build(train_rows)
        kb.save(kb_path)
    print(f"  KB size: {kb.size} entries")

    # Resume support — only skip commits that succeeded (not errors)
    done_shas = _load_done_shas(fold_idx, include_errors=False) if resume else set()
    pending = [r for r in test_rows if r["commit_sha"] not in done_shas]

    if limit:
        pending = pending[:limit]

    print(f"\n  Fold {fold_idx}: {len(test_rows)} test commits, "
          f"{len(done_shas)} done, {len(pending)} pending")

    if not pending:
        print("  Nothing to do — all commits already processed.")
        return []

    if dry_run:
        from tools import TOOL_SCHEMAS
        commit = pending[0]
        print("\n--- TARGET COMMIT ---")
        print(f"  SHA: {commit['commit_sha']}")
        print(f"  Message: {commit['masked_commit_message']}")
        print(f"  Human Label: {'Bug-Fix' if commit['human_label'] == 1 else 'Non-Bug-Fix'}")
        print(f"\n--- KB SEARCH PREVIEW ---")
        results = kb.search(
            commit["masked_commit_message"],
            commit.get("git_diff", ""),
            k=3,
            exclude_sha=commit["commit_sha"],
        )
        for i, r in enumerate(results, 1):
            print(f"  Ref {i}: [{r['label_str']}] {r['message'][:80]} "
                  f"(sim={r['similarity_score']:.3f})")
        print("\n--- AVAILABLE TOOLS ---")
        for t in TOOL_SCHEMAS:
            print(f"  • {t['name']}: {t['description'][:80]}...")
        return []

    # Pre-load model (only once)
    _load_model()

    # If resuming, keep only successful predictions (drop old errors for pending SHAs)
    if resume and pred_path.exists():
        pending_shas = {r["commit_sha"] for r in pending}
        kept_lines = []
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec["commit_sha"] not in pending_shas:
                            kept_lines.append(line.rstrip("\n"))
                    except (json.JSONDecodeError, KeyError):
                        pass
        with open(pred_path, "w", encoding="utf-8") as f:
            for l in kept_lines:
                f.write(l + "\n")
        mode = "a"
    else:
        mode = "w"

    results = []

    with open(pred_path, mode, encoding="utf-8") as fout:
        for i, commit in enumerate(pending, 1):
            print(
                f"  [{i:>3}/{len(pending)}] "
                f"{commit['repo_name']}/{commit['commit_sha'][:10]} ",
                end="", flush=True,
            )

            try:
                trace = run_agent(commit, kb)
                rec = trace.to_dict()
                label_str = (
                    "BUG" if trace.predicted_label == 1
                    else "NON" if trace.predicted_label == 0
                    else "ERR"
                )
                commit_tokens = trace.total_input_tokens + trace.total_output_tokens
                print(
                    f"→ {label_str} (conf={trace.confidence}, "
                    f"steps={trace.total_steps}, "
                    f"tools={len(trace.tool_calls)}, "
                    f"tokens={commit_tokens:,}, "
                    f"{trace.latency_seconds:.1f}s)"
                )
            except Exception as exc:
                rec = AgentTrace(
                    commit_sha=commit["commit_sha"],
                    repo_name=commit["repo_name"],
                    human_label=commit["human_label"],
                    masked_commit_message=commit["masked_commit_message"],
                    error=str(exc),
                ).to_dict()
                print(f"→ ERROR: {exc}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(rec)

    if not dry_run and pred_path.exists():
        fold_input = fold_output = 0
        n_preds = 0
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    fold_input += rec.get("total_input_tokens", 0)
                    fold_output += rec.get("total_output_tokens", 0)
                    n_preds += 1
        if n_preds:
            fold_total = fold_input + fold_output
            token_log = (
                f"Fold {fold_idx} token usage: "
                f"input={fold_input:,}, output={fold_output:,}, "
                f"total={fold_total:,} ({n_preds} commits, "
                f"avg={(fold_total / n_preds):,.0f}/commit)"
            )
            print(f"\n  {token_log}")
            (fold_dir / "token_usage.log").write_text(token_log + "\n", encoding="utf-8")

    return results


def run_metrics_only(fold_idx: int | None = None):
    """Recompute metrics from existing predictions."""
    if fold_idx is not None:
        folds_to_compute = [fold_idx]
    else:
        folds_to_compute = list(range(N_FOLDS))

    fold_metrics_list = []
    for fi in folds_to_compute:
        pred_path = _predictions_path(fi)
        if not pred_path.exists():
            print(f"  Fold {fi}: no predictions found, skipping.")
            continue
        metrics = write_fold_metrics(pred_path, fi)
        fold_metrics_list.append(metrics)

    if len(fold_metrics_list) > 1:
        summary = compute_cross_fold_metrics(fold_metrics_list)
        report = format_summary_report(summary)
        print(report)

        summary_path = DATA_DIR / "cross_fold_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summary_txt_path = DATA_DIR / "cross_fold_summary.txt"
        summary_txt_path.write_text(report, encoding="utf-8")
        print(f"  Summary → {summary_path}")
        print(f"  Report  → {summary_txt_path}")


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="Agentic Bug-Fix Commit Classification Experiment"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Run specific fold (0-4). Default: all folds.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from where it left off (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (overwrite previous results)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N test commits per fold")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sample prompts and KB results, no API calls")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics from existing predictions")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Only prepare data (merge datasets, create splits)")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    if args.prepare_data:
        save_merged_and_splits()
        return

    if args.metrics_only:
        run_metrics_only(args.fold)
        return

    # Load data
    rows, folds = load_splits()
    print(f"\nDataset: {len(rows)} commits, {N_FOLDS}-fold CV")
    print(f"Model: {MODEL_ID}")
    print(f"Resume: {args.resume}")
    print()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")

    # Determine which folds to run
    if args.fold is not None:
        fold_indices = [args.fold]
    else:
        fold_indices = list(range(N_FOLDS))

    # Run folds
    all_fold_metrics = []
    for fi in fold_indices:
        print(f"\n{'='*60}")
        print(f"  FOLD {fi}")
        print(f"{'='*60}")

        run_fold(
            fi, rows, folds,
            resume=args.resume,
            limit=args.limit,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            pred_path = _predictions_path(fi)
            if pred_path.exists():
                metrics = write_fold_metrics(pred_path, fi)
                all_fold_metrics.append(metrics)

    # Cross-fold summary
    if not args.dry_run and len(all_fold_metrics) > 1:
        summary = compute_cross_fold_metrics(all_fold_metrics)
        report = format_summary_report(summary)
        print(report)

        summary_path = DATA_DIR / "cross_fold_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        (DATA_DIR / "cross_fold_summary.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
