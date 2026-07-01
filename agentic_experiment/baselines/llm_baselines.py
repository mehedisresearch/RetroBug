"""Claude, GPT-4o, and CodeLlama single-shot LLM baselines."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_CLAUDE_EXP = Path(__file__).resolve().parent.parent.parent / "claude_message_experiment"
if str(_CLAUDE_EXP) not in sys.path:
    sys.path.append(str(_CLAUDE_EXP))

from parse_response import parse_model_output
from prompts import PromptConfig, build_user_prompt, get_system_prompt

from baselines.classical import BaselineClassifier
from baselines.gpt_prompts import GPT4O_SYSTEM_PROMPT, build_gpt4o_user_prompt


class ClaudeSingleShotBaseline(BaselineClassifier):
    """Single Claude API call — no agent, no KB, no tools."""

    def __init__(self, *, message_only: bool = True):
        self.message_only = message_only
        self.name = "claude_message" if message_only else "claude_message_diff"
        self.description = (
            "Claude Sonnet zero-shot, masked message only"
            if message_only
            else "Claude Sonnet zero-shot, masked message + git diff"
        )
        self.config = PromptConfig(
            with_reasoning=True,
            with_diff=not message_only,
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def fit(self, train_rows, kb=None):
        return self

    def predict(self, commit, kb=None) -> dict:
        client = self._get_client()
        system = get_system_prompt(self.config)
        user = build_user_prompt(
            commit["masked_commit_message"],
            config=self.config,
            git_diff=commit.get("git_diff", ""),
        )
        model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
        start = time.time()
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = round(time.time() - start, 2)
        text = "".join(b.text for b in resp.content if b.type == "text")
        label, conf, reasoning, err = parse_model_output(text, with_reasoning=True)
        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        return {
            "predicted_label": label,
            "confidence": conf,
            "reasoning": reasoning or text[:300],
            "error": err,
            "latency_seconds": latency,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "raw_output": text[:500],
        }


class Gpt4oSingleShotBaseline(BaselineClassifier):
    """Single GPT-4o API call — CCS-style prompt, masked message + full diff."""

    name = "gpt4o_message_diff"
    description = "GPT-4o zero-shot (CCS-style prompt), masked message + git diff"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package required for gpt4o_message_diff baseline. "
                    "Install with: pip install 'openai>=1.0.0'"
                ) from exc
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def fit(self, train_rows, kb=None):
        return self

    def predict(self, commit, kb=None) -> dict:
        client = self._get_client()
        system = GPT4O_SYSTEM_PROMPT
        user = build_gpt4o_user_prompt(
            commit["masked_commit_message"],
            commit.get("git_diff", ""),
        )
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        start = time.time()
        resp = None
        last_exc = None
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                break
            except Exception as exc:
                last_exc = exc
                err_text = str(exc).lower()
                if "429" in err_text or "rate_limit" in err_text:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise
        if resp is None:
            raise last_exc or RuntimeError("GPT-4o request failed after retries")
        latency = round(time.time() - start, 2)
        text = resp.choices[0].message.content or ""
        label, conf, reasoning, err = parse_model_output(text, with_reasoning=True)
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return {
            "predicted_label": label,
            "confidence": conf,
            "reasoning": reasoning or text[:300],
            "error": err,
            "latency_seconds": latency,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "raw_output": text[:500],
        }


class CodeLlamaSingleShotBaseline(BaselineClassifier):
    """Single CodeLlama call with message + diff JSON output."""

    name = "codellama_message_diff"
    description = "CodeLlama-13b single-shot, masked message + git diff"

    def fit(self, train_rows, kb=None):
        return self

    def predict(self, commit, kb=None) -> dict:
        import local_llm
        local_llm.set_backend("codellama")
        local_llm._load_model()

        system = (
            "Classify the commit as Bug-Fix (1) or Non-Bug-Fix (0). "
            "Reply with JSON only: "
            '{"label": 0 or 1, "confidence": 0-100, "reasoning": "..."}'
        )
        user = (
            f"Message: {commit['masked_commit_message']}\n\n"
            f"Diff:\n{commit.get('git_diff', '')[:6000]}"
        )
        start = time.time()
        resp = local_llm.generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
            max_new_tokens=512,
            temperature=0.0,
        )
        latency = round(time.time() - start, 2)
        text = resp.raw_output or resp.text
        label, conf, reasoning, err = parse_model_output(text, with_reasoning=True)
        return {
            "predicted_label": label,
            "confidence": conf,
            "reasoning": reasoning or text[:300],
            "error": err,
            "latency_seconds": latency,
            "raw_output": text[:500],
        }


LLM_BASELINES = [
    lambda: ClaudeSingleShotBaseline(message_only=True),
    lambda: ClaudeSingleShotBaseline(message_only=False),
    lambda: Gpt4oSingleShotBaseline(),
    lambda: CodeLlamaSingleShotBaseline(),
]
