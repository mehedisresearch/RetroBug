"""Prompts for simple 1-neighbor RAG: Forward CBR vs Backward CBR."""

from __future__ import annotations

from typing import Literal

CBRMode = Literal["backward", "forward"]

_BUGFIX_DEFINITION = """\
Definition of a bug-fix commit:
A bug-fix commit repairs an existing incorrect, failing, unsafe, or \
unintended behavior. It is NOT a bug-fix if it adds a feature, \
refactors code, updates documentation, reformats files, upgrades \
dependencies, or maintains tests without fixing a defect."""

CBR_SYSTEM_PROMPT = f"""\
You are an expert software engineering researcher specializing in \
classifying commits as Bug-Fix or Non-Bug-Fix.

You will receive:
1. One REFERENCE commit retrieved from a knowledge base (human-verified label).
2. One TARGET commit to classify (masked message and code diff).

{_BUGFIX_DEFINITION}

Reply with a single JSON object only — no markdown fences, no extra text."""


def format_reference_block(ref: dict) -> str:
    """Format the retrieved neighbor (same fields as agent KB search)."""
    parts = [
        f"Similarity score: {ref.get('similarity_score', 'n/a')}",
        f"Label: {ref['label_str']} (human-verified)",
        f"Repository: {ref['repo_name']}",
        f"Commit SHA: {ref['commit_sha']}",
        f"Message: {ref['message']}",
        f"Diff Summary: {ref['diff_summary']}",
    ]
    if ref.get("diff_snippet"):
        parts.append(f"Key Diff Lines:\n{ref['diff_snippet'][:600]}")
    if ref.get("fix_patterns"):
        parts.append(f"Detected Patterns: {', '.join(ref['fix_patterns'])}")
    return "\n".join(parts)


def format_target_block(commit: dict, *, diff_preview_chars: int = 3000) -> str:
    """Format the target commit (same preview length as the agent pipeline)."""
    parts = [
        f"Repository: {commit['repo_name']}",
        f"Commit SHA: {commit['commit_sha']}",
        f"Masked Commit Message: {commit['masked_commit_message']}",
    ]
    diff = commit.get("git_diff", "")
    if diff:
        preview = diff[:diff_preview_chars]
        if len(diff) > diff_preview_chars:
            preview += "\n... [diff truncated]"
        parts.append(f"\nCode Diff:\n{preview}")
    return "\n".join(parts)


def _backward_task(ref: dict) -> str:
    return (
        "=== YOUR TASK ===\n"
        f"1. Study the REFERENCE commit labeled '{ref['label_str']}'.\n"
        "2. Reason BACKWARD: WHY was this reference labeled that way? "
        "What specific evidence in its message and diff justified the label?\n"
        "3. Apply that reasoning to classify the TARGET commit.\n"
        "\n"
        'Reply with JSON: {"label": 0 or 1, "confidence": 0-100, '
        '"reasoning": "...", "backward_reasoning": "..."}'
    )


def _forward_task() -> str:
    return (
        "=== YOUR TASK ===\n"
        "Given this reference commit and its label, classify the TARGET commit.\n"
        "Use the reference as a similar labeled example.\n"
        "\n"
        'Reply with JSON: {"label": 0 or 1, "confidence": 0-100, "reasoning": "..."}'
    )


def build_user_prompt(
    commit: dict,
    reference: dict,
    mode: CBRMode,
    *,
    diff_preview_chars: int = 3000,
) -> str:
    """Build the single-turn user prompt (reference + target + task)."""
    parts = [
        "=== REFERENCE COMMIT (top-1 knowledge-base neighbor) ===",
        format_reference_block(reference),
        "",
        "=== TARGET COMMIT TO CLASSIFY ===",
        format_target_block(commit, diff_preview_chars=diff_preview_chars),
        "",
    ]
    if mode == "backward":
        parts.append(_backward_task(reference))
    else:
        parts.append(_forward_task())
    return "\n".join(parts)


def get_system_prompt() -> str:
    return CBR_SYSTEM_PROMPT
