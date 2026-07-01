"""Single-turn 1-neighbor RAG classifier (no tools, no agent loop)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from config import KB_TOP_K
from cbr_rag.prompts import CBRMode, build_user_prompt, get_system_prompt
from knowledge_base import KnowledgeBase

_CLAUDE_EXP = Path(__file__).resolve().parent.parent.parent / "claude_message_experiment"
if str(_CLAUDE_EXP) not in sys.path:
    sys.path.append(str(_CLAUDE_EXP))

from parse_response import parse_model_output  # noqa: E402


class CBRRagClassifier:
    """
    Retrieve one labeled neighbor, then one Claude call.

    Modes:
      backward — explain WHY the reference was labeled, then classify target
      forward  — given the reference, classify the target directly
    """

    def __init__(self, mode: CBRMode):
        if mode not in ("backward", "forward"):
            raise ValueError(f"mode must be 'backward' or 'forward', got {mode!r}")
        self.mode = mode
        self.name = f"{mode}_cbr_rag"
        self.description = (
            "Backward CBR RAG (1 neighbor, single turn): WHY was reference labeled?"
            if mode == "backward"
            else "Forward CBR RAG (1 neighbor, single turn): given reference, classify target"
        )
        self._client = None
        self.kb: KnowledgeBase | None = None

    def fit(self, train_rows, kb=None):
        self.kb = kb
        return self

    def _get_client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _retrieve(self, commit: dict) -> dict | None:
        if self.kb is None:
            raise ValueError("Knowledge base not set; call fit() with kb=...")
        results = self.kb.search(
            commit["masked_commit_message"],
            commit.get("git_diff", ""),
            k=KB_TOP_K,
            exclude_sha=commit["commit_sha"],
        )
        return results[0] if results else None

    def predict(self, commit, kb=None) -> dict:
        kb = kb or self.kb
        if kb is not None:
            self.kb = kb

        reference = self._retrieve(commit)
        if reference is None:
            return {
                "predicted_label": None,
                "confidence": None,
                "reasoning": "",
                "backward_reasoning": "",
                "error": "No KB neighbor found",
                "cbr_mode": self.mode,
                "kb_reference_sha": None,
                "kb_similarity": None,
            }

        client = self._get_client()
        system = get_system_prompt()
        user = build_user_prompt(commit, reference, self.mode)
        model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")

        start = time.time()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = round(time.time() - start, 2)
        text = "".join(b.text for b in resp.content if b.type == "text")

        label, conf, reasoning, err = parse_model_output(text, with_reasoning=True)
        backward_reasoning = ""
        if not err:
            try:
                obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
                backward_reasoning = str(obj.get("backward_reasoning", "")).strip()
            except (ValueError, json.JSONDecodeError):
                pass

        in_tok = getattr(resp.usage, "input_tokens", 0) or 0
        out_tok = getattr(resp.usage, "output_tokens", 0) or 0

        return {
            "predicted_label": label,
            "confidence": conf,
            "reasoning": reasoning or text[:500],
            "backward_reasoning": backward_reasoning,
            "error": err,
            "latency_seconds": latency,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "cbr_mode": self.mode,
            "kb_reference_sha": reference["commit_sha"],
            "kb_similarity": reference.get("similarity_score"),
            "raw_output": text[:800],
        }
