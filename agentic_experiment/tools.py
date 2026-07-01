"""
Tools available to the autonomous agent.

Each tool is defined as:
  - A schema (name, description, parameters) for the LLM
  - An execute function that runs the tool and returns a string result

The agent calls tools by name; the orchestrator dispatches to the execute function.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import MAX_KB_SEARCHES
from knowledge_base import KnowledgeBase, _summarize_diff, _detect_fix_patterns


TOOL_SCHEMAS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the knowledge base for the most similar previously-labeled commit. "
            "Returns the single best matching commit with its human-verified label, "
            "message, and diff snippet. Use this to find a reference case for "
            "backward reasoning — understanding WHY that commit was labeled "
            "as bug-fix or non-bug-fix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (commit message, keywords, or description of the change)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_diff_patterns",
        "description": (
            "Analyze the target commit's code diff for common bug-fix patterns "
            "(null checks, error handling, bounds checks, operator corrections, "
            "early returns, small targeted changes). Returns detected patterns "
            "and a structural summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Optional: what aspect to focus on (e.g., 'error handling', 'logic change')",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_diff_summary",
        "description": (
            "Get a structured summary of the diff: files changed, lines added/removed, "
            "and the most important changed lines. Cheaper than reading the full diff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check_behavioral_change",
        "description": (
            "Analyze whether the diff changes existing behavior (potential fix) "
            "or adds entirely new behavior (potential feature). Examines the "
            "ratio of modifications vs additions, and whether existing logic "
            "is altered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_message_intent",
        "description": (
            "Perform detailed analysis of the commit message for intent signals: "
            "fix keywords, feature keywords, refactoring indicators, and "
            "referenced issues/PRs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final classification. Call this when you have gathered "
            "enough evidence. You MUST call this tool to complete the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "integer",
                    "description": "1 for Bug-Fix, 0 for Non-Bug-Fix",
                    "enum": [0, 1],
                },
                "confidence": {
                    "type": "integer",
                    "description": "Confidence score 0-100",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Your full reasoning: (1) backward reasoning from reference examples, "
                        "(2) what evidence you gathered, (3) how it maps to the target commit, "
                        "(4) your conclusion."
                    ),
                },
                "backward_reasoning": {
                    "type": "string",
                    "description": (
                        "The backward reasoning you constructed from the KB reference: "
                        "WHY the reference was labeled the way it was."
                    ),
                },
            },
            "required": ["label", "confidence", "reasoning"],
        },
    },
]


def _extract_key_lines(diff: str, n: int = 20) -> str:
    lines = []
    for line in diff.split("\n"):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            lines.append(line)
        if len(lines) >= n:
            break
    return "\n".join(lines) if lines else "(no changes detected)"


class ToolExecutor:
    """Executes tools given the target commit context and KB."""

    def __init__(
        self,
        commit: dict,
        kb: KnowledgeBase,
        *,
        allowed_tools: set[str] | frozenset[str] | None = None,
        max_kb_searches: int | None = None,
    ):
        self.commit = commit
        self.kb = kb
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.max_kb_searches = max_kb_searches if max_kb_searches is not None else MAX_KB_SEARCHES
        self.call_log: list[dict] = []
        self.kb_search_count = 0

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch and execute a tool call. Returns result as string."""
        self.call_log.append({"tool": tool_name, "input": tool_input})

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return (
                f"Error: Tool '{tool_name}' is disabled in this ablation condition. "
                "Use only the tools available to you and call submit_answer when ready."
            )

        if tool_name == "search_knowledge_base":
            return self._search_kb(tool_input)
        elif tool_name == "analyze_diff_patterns":
            return self._analyze_diff_patterns(tool_input)
        elif tool_name == "get_diff_summary":
            return self._get_diff_summary(tool_input)
        elif tool_name == "check_behavioral_change":
            return self._check_behavioral_change(tool_input)
        elif tool_name == "analyze_message_intent":
            return self._analyze_message_intent(tool_input)
        elif tool_name == "submit_answer":
            return self._submit_answer(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _search_kb(self, params: dict) -> str:
        if self.max_kb_searches <= 0:
            return "Knowledge base search is disabled in this ablation condition."

        if self.kb_search_count >= self.max_kb_searches:
            return (
                f"Knowledge base search limit reached ({self.max_kb_searches} searches). "
                "Use the reference(s) you already retrieved plus the analysis tools, "
                "then call submit_answer with your best label."
            )

        self.kb_search_count += 1
        query = params.get("query", self.commit["masked_commit_message"])

        results = self.kb.search(
            query_message=query,
            query_diff=self.commit.get("git_diff", ""),
            k=1,
            exclude_sha=self.commit["commit_sha"],
        )

        if not results:
            return "No similar commits found in the knowledge base."

        r = results[0]
        output_parts = [
            f"--- MOST SIMILAR REFERENCE (similarity={r['similarity_score']:.3f}) ---",
            f"Label: {r['label_str']} (human-verified)",
            f"Repository: {r['repo_name']}",
            f"Message: {r['message']}",
            f"Diff Summary: {r['diff_summary']}",
        ]
        if r.get("diff_snippet"):
            snippet = r["diff_snippet"][:600]
            output_parts.append(f"Key Diff Lines:\n{snippet}")
        if r.get("fix_patterns"):
            output_parts.append(f"Detected Patterns: {', '.join(r['fix_patterns'])}")

        output_parts.append("")
        output_parts.append(
            "Now reason BACKWARD: WHY was this reference labeled as "
            f"'{r['label_str']}'? What evidence in its message and diff "
            "justified that label? Then apply the same reasoning to the target commit."
        )

        return "\n".join(output_parts)

    def _analyze_diff_patterns(self, params: dict) -> str:
        diff = self.commit.get("git_diff", "")
        if not diff:
            return "No diff available for this commit."

        patterns = _detect_fix_patterns(diff)
        summary = _summarize_diff(diff)
        key_lines = _extract_key_lines(diff, 25)

        parts = [
            f"Diff Summary: {summary}",
            f"Detected Fix Patterns: {', '.join(patterns) if patterns else 'none'}",
            "",
            "Key Changed Lines:",
            key_lines,
        ]

        focus = params.get("focus", "")
        if focus:
            parts.insert(0, f"[Focused analysis on: {focus}]")

        return "\n".join(parts)

    def _get_diff_summary(self, params: dict) -> str:
        diff = self.commit.get("git_diff", "")
        if not diff:
            return "No diff available for this commit."

        summary = _summarize_diff(diff)
        key_lines = _extract_key_lines(diff, 15)
        truncated = self.commit.get("diff_truncated", False)

        parts = [
            f"Summary: {summary}",
            f"Truncated: {truncated}",
            "",
            "Most Important Changes:",
            key_lines,
        ]
        return "\n".join(parts)

    def _check_behavioral_change(self, params: dict) -> str:
        diff = self.commit.get("git_diff", "")
        if not diff:
            return "No diff available."

        additions = 0
        deletions = 0
        modifications = 0
        new_functions = 0
        test_changes = 0

        lines = diff.split("\n")
        prev_deleted = False
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
                if prev_deleted:
                    modifications += 1
                if re.match(r"\+\s*(def |function |public |private |protected )", line):
                    new_functions += 1
                prev_deleted = False
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
                prev_deleted = True
            else:
                prev_deleted = False

            if "test" in line.lower() or "spec" in line.lower():
                test_changes += 1

        total_changes = additions + deletions
        mod_ratio = modifications / max(additions, 1)

        if mod_ratio > 0.5 and new_functions == 0:
            change_type = "MODIFIES existing behavior (likely fix or refactor)"
        elif new_functions > 0 and additions > deletions * 3:
            change_type = "ADDS new behavior (likely feature)"
        elif deletions > additions:
            change_type = "REMOVES behavior (cleanup or fix removing faulty code)"
        else:
            change_type = "MIXED (both adds and modifies)"

        return (
            f"Change Type: {change_type}\n"
            f"Additions: {additions} lines\n"
            f"Deletions: {deletions} lines\n"
            f"Modifications (line replacements): {modifications}\n"
            f"Modification Ratio: {mod_ratio:.2f}\n"
            f"New Functions/Methods: {new_functions}\n"
            f"Test-Related Changes: {test_changes} lines\n"
            f"Total Change Size: {total_changes} lines"
        )

    def _analyze_message_intent(self, params: dict) -> str:
        msg = self.commit["masked_commit_message"].lower()
        original = self.commit.get("commit_message", "").lower()

        fix_keywords = ["fix", "bug", "crash", "error", "patch", "resolve",
                        "repair", "correct", "issue", "fault", "defect",
                        "null", "npe", "race condition", "deadlock", "leak"]
        feature_keywords = ["add", "feat", "implement", "introduce", "new",
                           "support", "enable", "create"]
        refactor_keywords = ["refactor", "clean", "rename", "move", "extract",
                            "simplify", "restructure", "reorganize"]
        maintenance_keywords = ["update", "upgrade", "bump", "deps", "dependency",
                               "docs", "documentation", "readme", "changelog",
                               "format", "style", "lint"]

        found_fix = [kw for kw in fix_keywords if kw in msg or kw in original]
        found_feat = [kw for kw in feature_keywords if kw in msg or kw in original]
        found_refactor = [kw for kw in refactor_keywords if kw in msg or kw in original]
        found_maint = [kw for kw in maintenance_keywords if kw in msg or kw in original]

        issue_refs = re.findall(r"#(\d+)", self.commit.get("commit_message", ""))

        parts = [
            f"Masked Message: {self.commit['masked_commit_message']}",
            f"Original Message: {self.commit.get('commit_message', '(same)')}",
            "",
            f"Fix Keywords Found: {', '.join(found_fix) if found_fix else 'none'}",
            f"Feature Keywords Found: {', '.join(found_feat) if found_feat else 'none'}",
            f"Refactor Keywords Found: {', '.join(found_refactor) if found_refactor else 'none'}",
            f"Maintenance Keywords Found: {', '.join(found_maint) if found_maint else 'none'}",
            f"Issue References: {', '.join('#' + r for r in issue_refs) if issue_refs else 'none'}",
        ]

        if found_fix and not found_feat:
            parts.append("\nAssessment: Message strongly suggests bug-fix intent.")
        elif found_feat and not found_fix:
            parts.append("\nAssessment: Message strongly suggests feature/addition.")
        elif found_refactor:
            parts.append("\nAssessment: Message suggests refactoring (non-bug-fix).")
        elif found_maint:
            parts.append("\nAssessment: Message suggests maintenance (non-bug-fix).")
        else:
            parts.append("\nAssessment: Message is AMBIGUOUS — no clear intent keywords.")

        return "\n".join(parts)

    def _submit_answer(self, params: dict) -> str:
        return json.dumps(params)
