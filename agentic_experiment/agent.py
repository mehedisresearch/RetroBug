"""
Autonomous Agent for Bug-Fix Commit Classification.

Uses local Qwen3-Omni-30B-A3B-Instruct model with native tool-calling.

The agent receives a target commit + KB and autonomously:
1. Searches KB for similar commits
2. Reasons backward from their labels (WHY was this labeled X?)
3. Uses tools to gather evidence about the target
4. Produces a final classification with full reasoning trace
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Optional

from config import (
    AGENT_SYSTEM_PROMPT,
    MAX_AGENT_STEPS,
    MAX_KB_SEARCHES,
    MAX_TOKENS_PER_STEP,
    TEMPERATURE,
)
from knowledge_base import KnowledgeBase
from local_llm import (
    ModelResponse,
    ToolCall,
    format_assistant_with_tool_calls,
    format_tool_result,
    generate,
)
from tools import TOOL_SCHEMAS, ToolExecutor

if TYPE_CHECKING:
    from ablation.variants import AblationConfig


@dataclass
class AgentTrace:
    """Full trace of an agent classification run."""
    commit_sha: str
    repo_name: str
    human_label: int
    masked_commit_message: str

    predicted_label: Optional[int] = None
    confidence: Optional[int] = None
    reasoning: str = ""
    backward_reasoning: str = ""

    steps: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    thinking_trace: list[str] = field(default_factory=list)
    total_steps: int = 0
    latency_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    kb_references_used: list[dict] = field(default_factory=list)
    ablation_variant: Optional[str] = None

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_label(value) -> int:
    """Normalize label from int, string, or enum-like text to 0/1."""
    if value is None:
        raise ValueError("missing label")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"invalid label int: {value}")
    text = str(value).strip().lower()
    if text in ("1", "bug-fix", "bug fix", "bugfix", "bug_fix"):
        return 1
    if text in ("0", "non-bug-fix", "non bug-fix", "non bug fix", "non-bug", "nonbug"):
        return 0
    raise ValueError(f"invalid label: {value}")


def _parse_submit_answer(result_str: str) -> dict:
    answer = json.loads(result_str)
    answer["label"] = _parse_label(answer.get("label"))
    answer["confidence"] = int(answer.get("confidence", 0))
    return answer


def _build_user_message(
    commit: dict,
    task_instructions: tuple[str, ...] | None = None,
) -> str:
    """Build the initial user message with the target commit."""
    parts = [
        "=== TARGET COMMIT TO CLASSIFY ===",
        f"Repository: {commit['repo_name']}",
        f"Commit SHA: {commit['commit_sha']}",
        f"Masked Commit Message: {commit['masked_commit_message']}",
    ]

    if commit.get("git_diff"):
        diff_preview = commit["git_diff"][:3000]
        if len(commit["git_diff"]) > 3000:
            diff_preview += "\n... [diff continues, use get_diff_summary or analyze_diff_patterns for full analysis]"
        parts.append(f"\nCode Diff (preview):\n{diff_preview}")

    parts.append("\n=== YOUR TASK ===")
    instructions = task_instructions or (
        "1. Use `search_knowledge_base` to find similar previously-labeled commits.",
        "2. Study the references. Reason BACKWARD: WHY were they labeled Bug-Fix or Non-Bug-Fix?",
        "3. Use other tools to gather evidence about the target if needed.",
        "4. Call `submit_answer` with your classification (label, confidence, reasoning).",
        f"5. You may search the knowledge base at most {MAX_KB_SEARCHES} times — then decide and submit.",
    )
    parts.extend(instructions)

    return "\n".join(parts)


def _active_tool_schemas(allowed_tools: frozenset[str] | None) -> list[dict]:
    if allowed_tools is None:
        return TOOL_SCHEMAS
    return [schema for schema in TOOL_SCHEMAS if schema["name"] in allowed_tools]


def _submit_only_tools(allowed_tools: frozenset[str] | None = None) -> list[dict]:
    schemas = _active_tool_schemas(allowed_tools)
    return [schema for schema in schemas if schema["name"] == "submit_answer"]


def _nudge_message(
    step_idx: int,
    kb_searches: int,
    *,
    max_steps: int = MAX_AGENT_STEPS,
    max_kb_searches: int = MAX_KB_SEARCHES,
) -> str | None:
    """Return a user nudge when the agent should stop searching and submit."""
    steps_left = max_steps - step_idx - 1

    if steps_left <= 1:
        return (
            "FINAL STEP: You MUST call `submit_answer` now with label (0 or 1), "
            "confidence (0-100), reasoning, and backward_reasoning. "
            "Do not call any other tools."
        )

    if kb_searches >= max_kb_searches:
        return (
            f"You have used all {max_kb_searches} knowledge base searches. "
            "Do not search again. Use the evidence you already collected and "
            "call `submit_answer` now."
        )

    if step_idx >= 5:
        return (
            "You have gathered enough evidence. Call `submit_answer` now with "
            "your label (0 or 1), confidence, and reasoning."
        )

    if step_idx >= 3 and kb_searches >= 1:
        return (
            "You already have a KB reference and analysis results. "
            "If you need one more check, use analysis tools — otherwise "
            "call `submit_answer` now."
        )

    return None


def _force_final_submission(
    messages: list[dict],
    allowed_tools: frozenset[str] | None = None,
) -> ModelResponse:
    """Last-resort call that only allows submit_answer."""
    return generate(
        messages=messages + [{
            "role": "user",
            "content": (
                "Time is up. You MUST call `submit_answer` immediately with your "
                "best label (0 or 1), confidence (0-100), reasoning, and "
                "backward_reasoning based on all evidence gathered so far."
            ),
        }],
        tools=_submit_only_tools(allowed_tools),
        max_new_tokens=MAX_TOKENS_PER_STEP,
        temperature=TEMPERATURE,
    )


def _accumulate_tokens(trace: AgentTrace, response: ModelResponse):
    trace.total_input_tokens += response.input_tokens
    trace.total_output_tokens += response.output_tokens


def run_agent(
    commit: dict,
    kb: KnowledgeBase,
    ablation: "AblationConfig | None" = None,
) -> AgentTrace:
    """
    Run the autonomous agent on a single commit.
    Returns the full trace including all steps, tool calls, and reasoning.

    When `ablation` is provided, tool availability, step limits, and prompts
    are adjusted for that ablation condition (main pipeline unchanged).
    """
    allowed_tools = ablation.allowed_tools if ablation else None
    max_steps = ablation.max_steps if ablation else MAX_AGENT_STEPS
    max_kb_searches = ablation.max_kb_searches if ablation else MAX_KB_SEARCHES
    system_prompt = ablation.system_prompt if ablation else AGENT_SYSTEM_PROMPT
    task_instructions = ablation.task_instructions if ablation else None
    single_turn = ablation.single_turn if ablation else False
    tool_schemas = _active_tool_schemas(allowed_tools)

    trace = AgentTrace(
        commit_sha=commit["commit_sha"],
        repo_name=commit["repo_name"],
        human_label=commit["human_label"],
        masked_commit_message=commit["masked_commit_message"],
        ablation_variant=ablation.name if ablation else None,
    )

    executor = ToolExecutor(
        commit,
        kb,
        allowed_tools=allowed_tools,
        max_kb_searches=max_kb_searches,
    )
    user_message = _build_user_message(commit, task_instructions)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start_time = time.time()
    answer_submitted = False
    last_response: ModelResponse | None = None

    try:
        for step_idx in range(max_steps):
            if single_turn and max_steps >= 2:
                tools = (
                    tool_schemas if step_idx == 0
                    else _submit_only_tools(allowed_tools)
                )
            elif single_turn:
                tools = tool_schemas
            elif step_idx >= max_steps - 1:
                tools = _submit_only_tools(allowed_tools)
            else:
                tools = tool_schemas

            response: ModelResponse = generate(
                messages=messages,
                tools=tools,
                max_new_tokens=MAX_TOKENS_PER_STEP,
                temperature=TEMPERATURE,
            )
            last_response = response
            _accumulate_tokens(trace, response)

            trace.total_steps = step_idx + 1

            if response.thinking:
                trace.thinking_trace.append(response.thinking)

            step_record = {
                "step": step_idx,
                "thinking": response.thinking[:500] if response.thinking else "",
                "text": response.text[:500] if response.text else "",
                "tool_calls": [],
            }

            if response.has_tool_calls:
                tool_results = []

                for tc in response.tool_calls:
                    result_str = executor.execute(tc.name, tc.arguments)

                    step_record["tool_calls"].append({
                        "tool": tc.name,
                        "input": tc.arguments,
                        "result_preview": result_str[:300],
                    })

                    trace.tool_calls.append({
                        "step": step_idx,
                        "tool": tc.name,
                        "input": tc.arguments,
                        "result_preview": result_str[:500],
                    })

                    if tc.name == "submit_answer":
                        answer_submitted = True
                        try:
                            answer = _parse_submit_answer(result_str)
                            trace.predicted_label = answer["label"]
                            trace.confidence = answer["confidence"]
                            trace.reasoning = answer.get("reasoning", "")
                            trace.backward_reasoning = answer.get("backward_reasoning", "")
                            trace.error = None
                        except (json.JSONDecodeError, ValueError, TypeError) as exc:
                            trace.predicted_label = None
                            trace.error = f"Failed to parse submit_answer: {exc}; {result_str[:200]}"

                    tool_results.append(format_tool_result(tc.name, result_str))

                call_ids = getattr(response, "_call_ids", None)
                messages.append(format_assistant_with_tool_calls(
                    response.text, response.tool_calls, call_ids
                ))
                for tr in tool_results:
                    messages.append(tr)

            else:
                messages.append({"role": "assistant", "content": response.text})

                if not answer_submitted and not single_turn:
                    nudge = _nudge_message(
                        step_idx,
                        executor.kb_search_count,
                        max_steps=max_steps,
                        max_kb_searches=max_kb_searches,
                    )
                    if nudge:
                        messages.append({"role": "user", "content": nudge})
                    else:
                        messages.append({
                            "role": "user",
                            "content": (
                                "You haven't called `submit_answer` yet. "
                                "Please call it now with your label (0 or 1), "
                                "confidence (0-100), and reasoning."
                            ),
                        })

            trace.steps.append(step_record)

            if answer_submitted:
                break

            if not single_turn:
                post_step_nudge = _nudge_message(
                    step_idx,
                    executor.kb_search_count,
                    max_steps=max_steps,
                    max_kb_searches=max_kb_searches,
                )
                if post_step_nudge and not answer_submitted:
                    messages.append({"role": "user", "content": post_step_nudge})

        if not answer_submitted:
            forced = _force_final_submission(messages, allowed_tools)
            last_response = forced
            _accumulate_tokens(trace, forced)
            trace.total_steps += 1

            if forced.has_tool_calls:
                for tc in forced.tool_calls:
                    if tc.name != "submit_answer":
                        continue
                    result_str = executor.execute(tc.name, tc.arguments)
                    trace.tool_calls.append({
                        "step": trace.total_steps - 1,
                        "tool": tc.name,
                        "input": tc.arguments,
                        "result_preview": result_str[:500],
                    })
                    try:
                        answer = _parse_submit_answer(result_str)
                        trace.predicted_label = answer["label"]
                        trace.confidence = answer["confidence"]
                        trace.reasoning = answer.get("reasoning", "")
                        trace.backward_reasoning = answer.get("backward_reasoning", "")
                        trace.error = None
                        answer_submitted = True
                    except (json.JSONDecodeError, ValueError, TypeError) as exc:
                        trace.error = f"Failed to parse forced submit_answer: {exc}; {result_str[:200]}"

        if not answer_submitted:
            trace.error = f"Agent did not submit answer within {max_steps} steps"
            if last_response is not None:
                _try_fallback_extraction(trace, last_response)
            _try_fallback_from_trace(trace)

    except Exception as exc:
        trace.error = str(exc)

    trace.latency_seconds = round(time.time() - start_time, 2)

    # Record which KB references were used
    for call in executor.call_log:
        if call["tool"] == "search_knowledge_base":
            results = kb.search(
                query_message=call["input"].get("query", commit["masked_commit_message"]),
                query_diff=commit.get("git_diff", ""),
                k=call["input"].get("k", 1),
                exclude_sha=commit["commit_sha"],
            )
            trace.kb_references_used = results
            break

    return trace


def _try_fallback_extraction(trace: AgentTrace, last_response: ModelResponse):
    """
    If the agent never called submit_answer, try to extract a label
    from its text output as a fallback.
    """
    text = (last_response.text + " " + last_response.thinking).lower()

    if any(p in text for p in ("bug-fix", "bug fix", "label: 1", '"label": 1', "classified as bug")):
        trace.predicted_label = 1
        trace.reasoning = f"[FALLBACK] Extracted from agent text: {last_response.text[:300]}"
        trace.error = None
    elif any(p in text for p in (
        "non-bug", "not a bug", "label: 0", '"label": 0',
        "non bug-fix", "non bug fix", "classified as non",
    )):
        trace.predicted_label = 0
        trace.reasoning = f"[FALLBACK] Extracted from agent text: {last_response.text[:300]}"
        trace.error = None


def _try_fallback_from_trace(trace: AgentTrace):
    """Scan all step text for an explicit classification decision."""
    if trace.predicted_label is not None:
        return

    combined = " ".join(
        step.get("text", "") for step in trace.steps
    ).lower()

    bug_signals = (
        "this is a bug-fix", "classified as bug-fix", "label: 1",
        "conclusion: bug-fix", "therefore bug-fix", "likely bug-fix",
    )
    non_signals = (
        "this is a non-bug", "not a bug-fix", "non-bug-fix",
        "label: 0", "conclusion: non-bug", "therefore non-bug",
        "likely non-bug", "maintenance change", "not fixing a defect",
    )

    bug_hits = sum(1 for s in bug_signals if s in combined)
    non_hits = sum(1 for s in non_signals if s in combined)

    if non_hits > bug_hits and non_hits > 0:
        trace.predicted_label = 0
        trace.reasoning = "[FALLBACK] Inferred Non-Bug-Fix from agent step text."
        trace.error = None
    elif bug_hits > non_hits and bug_hits > 0:
        trace.predicted_label = 1
        trace.reasoning = "[FALLBACK] Inferred Bug-Fix from agent step text."
        trace.error = None
