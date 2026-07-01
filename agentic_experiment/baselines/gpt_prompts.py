"""
CCS-style prompts for GPT-4o bug-fix classification baseline.

Adapted from the commit-classification prompt in the CCS paper: binary
Bug-Fix vs Non-Bug-Fix using masked commit message and full git diff.
"""

from __future__ import annotations

# Mirrors the CCS paper structure; categories reduced to two classes.
GPT4O_SYSTEM_PROMPT = """\
You are a commit classifier based on commit message and code diff.\
"""

GPT4O_USER_PROMPT_TEMPLATE = """\
Please classify the given commit into one of two categories: Bug-Fix or \
Non-Bug-Fix. The definitions of each category are as follows:

Bug-Fix:
A commit that repairs existing incorrect, failing, unsafe, or unintended \
behavior in the software. This includes fixing crashes, logical errors, \
incorrect outputs, performance defects, security vulnerabilities, race \
conditions, or regressions introduced by earlier changes.

Non-Bug-Fix:
A commit that does not repair a software defect. This includes adding \
features, refactoring code, updating documentation, reformatting or \
style-only changes, upgrading dependencies, build or CI configuration \
changes, and test maintenance that updates tests without fixing a \
production bug.

Task:
Analyze both the commit message and the code diff. Return 1 if the commit \
is a Bug-Fix, or 0 if it is a Non-Bug-Fix. Provide a confidence score \
from 0 to 100 and brief reasoning grounded in both the message and the diff.

Output Format (strict JSON only, no markdown fences):
{{"label": <0 or 1>, "confidence": <0-100>, "reasoning": "<2-4 sentences>"}}

Rules:
- label must be 0 (Non-Bug-Fix) or 1 (Bug-Fix)
- confidence must be an integer from 0 to 100
- reasoning must cite evidence from both the message and the diff
- output only the JSON object, nothing else

The given commit message:
{commit_message}

The given commit diff:
{commit_diff}\
"""


def build_gpt4o_user_prompt(masked_message: str, git_diff: str) -> str:
    """Build the CCS-style user prompt with masked message and full diff."""
    return GPT4O_USER_PROMPT_TEMPLATE.format(
        commit_message=masked_message.strip(),
        commit_diff=git_diff.strip() or "(empty diff)",
    )
