"""
Knowledge Base: Build a semantic embedding index over labeled commits (train set per fold).
Uses all-MiniLM-L6-v2 sentence-transformer for similarity search on message + diff summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    DIFF_SNIPPET_LINES,
    KB_TOP_K,
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def _extract_diff_snippet(diff: str, max_lines: int = DIFF_SNIPPET_LINES) -> str:
    """Extract the most informative changed lines from a diff."""
    important = []
    for line in diff.split("\n"):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            important.append(line)
        if len(important) >= max_lines:
            break
    return "\n".join(important)


def _summarize_diff(diff: str) -> str:
    """Produce a compact textual summary of the diff."""
    if not diff:
        return "(no diff)"
    files = set()
    additions = 0
    deletions = 0
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) > 1:
                files.add(parts[-1])
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    file_list = ", ".join(sorted(files)[:5])
    if len(files) > 5:
        file_list += f" (+{len(files) - 5} more)"
    return f"{len(files)} files, +{additions} -{deletions} lines. Files: {file_list}"


def _detect_fix_patterns(diff: str) -> list[str]:
    """Detect common bug-fix patterns in the diff."""
    patterns = []
    lower = diff.lower()
    if re.search(r"\+.*\bif\b.*\bnull\b|\+.*\bif\b.*\bnone\b|\+.*!=\s*null", lower):
        patterns.append("null_check_added")
    if re.search(r"\+.*\btry\b|\+.*\bcatch\b|\+.*\bexcept\b", lower):
        patterns.append("error_handling_added")
    if re.search(r"\+.*\bif\b.*\blen\b|\+.*\bif\b.*\bempty\b|\+.*\bif\b.*\.size", lower):
        patterns.append("bounds_check_added")
    if re.search(r"-.*\b(>|<|>=|<=|==|!=)\b.*\n\+.*\b(>|<|>=|<=|==|!=)\b", diff):
        patterns.append("comparison_operator_changed")
    if re.search(r"\+.*\breturn\b.*\bearly\b|\+.*\breturn\b.*\n.*\bif\b", lower):
        patterns.append("early_return_added")
    if re.search(r"-.*\n\+.*", diff) and _additions_count(diff) < 5 and _deletions_count(diff) < 5:
        patterns.append("small_targeted_change")
    return patterns


def _additions_count(diff: str) -> int:
    return sum(1 for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++"))


def _deletions_count(diff: str) -> int:
    return sum(1 for l in diff.split("\n") if l.startswith("-") and not l.startswith("---"))


def _build_search_text(message: str, diff: str) -> str:
    """Combine message + diff summary into a single text for embedding."""
    diff_summary = _summarize_diff(diff) if diff else ""
    return f"{message} {diff_summary}".strip()


class KnowledgeBase:
    """Semantic embedding-based knowledge base for commit similarity search."""

    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.entries: list[dict] = []

    def build(self, train_rows: list[dict]):
        """Build the KB from training rows using sentence-transformer embeddings."""
        self.entries = []
        texts = []

        for row in train_rows:
            diff = row.get("git_diff", "")
            diff_summary = _summarize_diff(diff)
            diff_snippet = _extract_diff_snippet(diff)
            fix_patterns = _detect_fix_patterns(diff)

            entry = {
                "commit_sha": row["commit_sha"],
                "repo_name": row["repo_name"],
                "message": row["masked_commit_message"],
                "diff_summary": diff_summary,
                "diff_snippet": diff_snippet,
                "fix_patterns": fix_patterns,
                "human_label": row["human_label"],
                "label_str": "Bug-Fix" if row["human_label"] == 1 else "Non-Bug-Fix",
            }
            self.entries.append(entry)
            texts.append(_build_search_text(row["masked_commit_message"], diff))

        model = _get_embed_model()
        print(f"    Encoding {len(texts)} commits with {EMBEDDING_MODEL_NAME}...")
        self.embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )

    def search(
        self,
        query_message: str,
        query_diff: str = "",
        k: int = KB_TOP_K,
        exclude_sha: str | None = None,
    ) -> list[dict]:
        """Find top-k similar commits from the KB using cosine similarity."""
        if self.embeddings is None or len(self.entries) == 0:
            return []

        model = _get_embed_model()
        query_text = _build_search_text(query_message, query_diff)
        query_emb = model.encode(
            [query_text],
            normalize_embeddings=True,
        )

        # Cosine similarity (embeddings are already L2-normalized)
        scores = (query_emb @ self.embeddings.T).flatten()

        if exclude_sha:
            for i, entry in enumerate(self.entries):
                if entry["commit_sha"] == exclude_sha:
                    scores[i] = -1.0

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                **self.entries[idx],
                "similarity_score": round(float(scores[idx]), 4),
            })

        return results

    def save(self, path: Path):
        """Save KB to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embeddings": self.embeddings,
            "entries": self.entries,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
        joblib.dump(payload, path)

    def load(self, path: Path):
        """Load KB from disk."""
        payload = joblib.load(path)
        self.embeddings = payload["embeddings"]
        self.entries = payload["entries"]

    @property
    def size(self) -> int:
        return len(self.entries)
