"""
Load and merge both datasets, create stratified 5-fold CV splits.

Each commit record contains:
  - commit_sha, repo_name, commit_url
  - commit_message, masked_commit_message
  - git_diff (truncated to DIFF_CHAR_LIMIT)
  - human_label (0 or 1)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_DIR,
    DATASET_PATHS,
    DIFF_CHAR_LIMIT,
    N_FOLDS,
    RANDOM_SEED,
)

csv.field_size_limit(sys.maxsize)

SPLITS_FILE = DATA_DIR / "cv_splits.json"
MERGED_FILE = DATA_DIR / "merged_dataset.jsonl"


def _repo_from_url(url: str) -> str:
    if not url or "github.com/" not in url:
        return ""
    parts = url.split("github.com/", 1)[-1].strip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return ""


def _truncate_diff(diff: str, limit: int = DIFF_CHAR_LIMIT) -> tuple[str, bool]:
    if len(diff) <= limit:
        return diff, False
    cutoff = diff.rfind("\n@@", 0, limit)
    if cutoff == -1:
        cutoff = limit
    return diff[:cutoff], True


def load_single_csv(path: Path) -> list[dict]:
    rows = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            msg = (row.get("masked_commit_message") or row.get("commit_message") or "").strip()
            if not msg:
                continue
            try:
                label = int(float(row["human_label"]))
            except (ValueError, TypeError):
                skipped += 1
                continue
            if label not in (0, 1):
                skipped += 1
                continue
            sha = (row.get("sha") or "").strip()
            if not sha or len(sha) < 7:
                skipped += 1
                continue
            raw_diff = (row.get("git_diff") or "").strip()
            diff, truncated = _truncate_diff(raw_diff)
            rec = {
                "commit_sha": sha,
                "repo_name": _repo_from_url(row.get("commit_url", "")),
                "commit_url": row.get("commit_url", "").strip(),
                "commit_message": (row.get("commit_message") or "").strip(),
                "masked_commit_message": msg,
                "git_diff": diff,
                "diff_truncated": truncated,
                "human_label": label,
            }
            rows.append(rec)
    if skipped:
        print(f"  Warning: skipped {skipped} malformed rows in {path.name}")
    return rows


def load_merged_dataset() -> list[dict]:
    """Load and merge both datasets, deduplicating by commit_sha."""
    all_rows = []
    seen_shas = set()

    for path in DATASET_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        rows = load_single_csv(path)
        for row in rows:
            if row["commit_sha"] not in seen_shas:
                seen_shas.add(row["commit_sha"])
                all_rows.append(row)

    all_rows.sort(key=lambda r: r["commit_sha"])
    return all_rows


def create_cv_splits(rows: list[dict]) -> list[dict]:
    """
    Create stratified 5-fold CV splits.
    Returns list of fold definitions: [{train_shas: [...], test_shas: [...]}]
    """
    labels = np.array([r["human_label"] for r in rows])
    shas = np.array([r["commit_sha"] for r in rows])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(shas, labels)):
        fold = {
            "fold": fold_idx,
            "train_shas": shas[train_idx].tolist(),
            "test_shas": shas[test_idx].tolist(),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_bugfix": int(labels[train_idx].sum()),
            "test_bugfix": int(labels[test_idx].sum()),
        }
        folds.append(fold)

    return folds


def save_merged_and_splits():
    """Load data, create splits, and save to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_merged_dataset()
    print(f"Merged dataset: {len(rows)} commits")
    print(f"  Bug-Fix: {sum(1 for r in rows if r['human_label'] == 1)}")
    print(f"  Non-Bug-Fix: {sum(1 for r in rows if r['human_label'] == 0)}")

    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved → {MERGED_FILE}")

    folds = create_cv_splits(rows)
    with open(SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    print(f"\n{N_FOLDS}-Fold CV splits:")
    for fold in folds:
        print(
            f"  Fold {fold['fold']}: "
            f"train={fold['train_size']} ({fold['train_bugfix']} bug-fix) | "
            f"test={fold['test_size']} ({fold['test_bugfix']} bug-fix)"
        )
    print(f"  Saved → {SPLITS_FILE}")

    return rows, folds


def load_splits() -> tuple[list[dict], list[dict]]:
    """Load previously saved merged dataset and CV splits."""
    if not MERGED_FILE.exists() or not SPLITS_FILE.exists():
        return save_merged_and_splits()

    rows = []
    with open(MERGED_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with open(SPLITS_FILE, encoding="utf-8") as f:
        folds = json.load(f)

    return rows, folds


def get_fold_data(
    rows: list[dict], folds: list[dict], fold_idx: int
) -> tuple[list[dict], list[dict]]:
    """Get train and test rows for a specific fold."""
    fold = folds[fold_idx]
    train_shas = set(fold["train_shas"])
    test_shas = set(fold["test_shas"])

    train_rows = [r for r in rows if r["commit_sha"] in train_shas]
    test_rows = [r for r in rows if r["commit_sha"] in test_shas]

    return train_rows, test_rows


if __name__ == "__main__":
    save_merged_and_splits()
