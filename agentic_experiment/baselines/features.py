"""Feature extraction for classical baselines."""

from __future__ import annotations

import re

FIX_KEYWORDS = re.compile(
    r"\b(fix|bug|patch|hotfix|regression|crash|error|issue|broken|defect)\b",
    re.I,
)
MAINT_KEYWORDS = re.compile(
    r"\b(docs|doc|readme|refactor|style|lint|format|version|bump|update|ci|test)\b",
    re.I,
)


def diff_stats(diff: str) -> dict[str, float]:
    if not diff:
        return {
            "n_files": 0,
            "additions": 0,
            "deletions": 0,
            "total_changed": 0,
            "test_lines": 0,
            "test_ratio": 0.0,
        }
    files = set()
    additions = deletions = test_lines = 0
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) > 1:
                files.add(parts[-1])
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
            if _is_test_path(line):
                test_lines += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            if _is_test_path(line):
                test_lines += 1
    total = additions + deletions
    return {
        "n_files": float(len(files)),
        "additions": float(additions),
        "deletions": float(deletions),
        "total_changed": float(total),
        "test_lines": float(test_lines),
        "test_ratio": test_lines / total if total else 0.0,
    }


def _is_test_path(line: str) -> bool:
    lower = line.lower()
    return any(k in lower for k in ("test", "spec", "__tests__", "/tests/"))


def message_features(message: str) -> dict[str, float]:
    msg = message or ""
    return {
        "msg_len": float(len(msg)),
        "msg_words": float(len(msg.split())),
        "fix_kw": float(len(FIX_KEYWORDS.findall(msg))),
        "maint_kw": float(len(MAINT_KEYWORDS.findall(msg))),
    }


def hybrid_feature_vector(message: str, diff: str) -> dict[str, float]:
    return {**message_features(message), **diff_stats(diff)}
