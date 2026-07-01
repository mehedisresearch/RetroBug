"""
Prompt templates for Claude commit classification experiments.

Modes:
  - label_only: masked message → "1 85"
  - reasoning: masked message → JSON with reasoning
  - message_diff_reasoning: masked message + git diff → JSON with reasoning
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """\
You are an expert developer and software engineering researcher.
Given a commit message, your task is to classify the commit as a
bug-fix or not.\
"""

USER_PROMPT_TEMPLATE = """\
Definition of a bug-fix commit:
A bug-fix commit refers to a change intended to correct an error, flaw,
or unintended behavior in the software that causes it to produce incor-
rect or unexpected results, or to fail in some way. This includes fixing
crashes, logical errors, performance issues, or addressing unintended
behavior.

Task: Analyze the commit message as input, return 1 if it is a bug-fix
commit. Otherwise, return 0 if it is not a bug-fix commit. This should
be followed by a second number in the range 0-100 representing how
confident you are in identifying a bug-fix commit.

Output Format (strict):
• The output must be a single integer, either 1 or 0, followed by
a numeric score between 0 and 100, indicating your confidence
in classifying the commit. A score of 0 represents the lowest
confidence, and 100 represents the highest.
• No explanation, no rationale, and no additional output

Input Commit Message :
{message}\
"""

SYSTEM_PROMPT_WITH_REASONING = """\
You are an expert developer and software engineering researcher.
Given a commit message, classify it as a bug-fix or not and explain
your reasoning based only on the message text.\
"""

USER_PROMPT_WITH_REASONING_TEMPLATE = """\
Definition of a bug-fix commit:
A bug-fix commit refers to a change intended to correct an error, flaw,
or unintended behavior in the software that causes it to produce incor-
rect or unexpected results, or to fail in some way. This includes fixing
crashes, logical errors, performance issues, or addressing unintended
behavior.

Task: Analyze the commit message below. Decide whether it is a bug-fix
commit (1) or not (0). Provide a confidence score from 0 to 100, and
brief reasoning grounded in wording from the message.

Output Format (strict JSON only, no markdown fences):
{{"label": <0 or 1>, "confidence": <0-100>, "reasoning": "<1-3 sentences>"}}

Rules:
• label must be 0 or 1
• confidence must be an integer 0-100
• reasoning must cite specific phrases or patterns from the message
• output only the JSON object, nothing else

Input Commit Message :
{message}\
"""

SYSTEM_PROMPT_MESSAGE_DIFF_REASONING = """\
You are an expert developer and software engineering researcher.
Given a masked commit message and the git diff, classify the commit as
a bug-fix or not and explain your reasoning using both inputs.\
"""

USER_PROMPT_MESSAGE_DIFF_REASONING_TEMPLATE = """\
Definition of a bug-fix commit:
A bug-fix commit refers to a change intended to correct an error, flaw,
or unintended behavior in the software that causes it to produce incor-
rect or unexpected results, or to fail in some way. This includes fixing
crashes, logical errors, performance issues, or addressing unintended
behavior.

Task: Analyze the masked commit message and git diff below. Decide
whether this is a bug-fix commit (1) or not (0). Provide a confidence
score from 0 to 100 and brief reasoning grounded in both the message
wording and concrete code changes in the diff.

Output Format (strict JSON only, no markdown fences):
{{"label": <0 or 1>, "confidence": <0-100>, "reasoning": "<2-4 sentences>"}}

Rules:
• label must be 0 or 1
• confidence must be an integer 0-100
• reasoning must reference both message cues and diff patterns (e.g.
  guard clauses, null checks, corrected logic, test fixes for failures)
• output only the JSON object, nothing else

Input Commit Message :
{message}

Git Diff :
{diff}\
"""


@dataclass(frozen=True)
class PromptConfig:
    with_reasoning: bool = False
    with_diff: bool = False

    @property
    def slug(self) -> str:
        if self.with_diff and self.with_reasoning:
            return "message_diff_reasoning"
        if self.with_reasoning:
            return "reasoning"
        return "label_only"

    @property
    def title(self) -> str:
        if self.with_diff and self.with_reasoning:
            return "Claude message + diff + reasoning"
        if self.with_reasoning:
            return "Claude message-only + reasoning"
        return "Claude message-only"

    @property
    def uses_json_output(self) -> bool:
        return self.with_reasoning or self.with_diff


def get_system_prompt(config: PromptConfig) -> str:
    if config.with_diff and config.with_reasoning:
        return SYSTEM_PROMPT_MESSAGE_DIFF_REASONING
    if config.with_reasoning:
        return SYSTEM_PROMPT_WITH_REASONING
    return SYSTEM_PROMPT


def build_user_prompt(
    masked_message: str,
    *,
    config: PromptConfig,
    git_diff: str = "",
) -> str:
    msg = masked_message.strip()
    if config.with_diff and config.with_reasoning:
        return USER_PROMPT_MESSAGE_DIFF_REASONING_TEMPLATE.format(
            message=msg,
            diff=git_diff.strip() or "(empty diff)",
        )
    if config.with_reasoning:
        return USER_PROMPT_WITH_REASONING_TEMPLATE.format(message=msg)
    return USER_PROMPT_TEMPLATE.format(message=msg)
