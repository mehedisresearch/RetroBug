"""Parse Claude responses for label-only and reasoning variants."""

from __future__ import annotations

import json
import re

LABEL_ONLY_RE = re.compile(r"^\s*([01])\s+(\d{1,3})\s*\.?\s*$")
JSON_RE = re.compile(r"\{[\s\S]*\}")


def _clamp_confidence(value: int) -> int:
    return max(0, min(100, value))


def parse_label_only(text: str) -> tuple[int | None, int | None, str | None, str | None]:
    """Parse '1 85'. Returns (label, confidence, reasoning, error)."""
    if not text:
        return None, None, None, "empty response"
    cleaned = text.strip().splitlines()[0].strip()
    m = LABEL_ONLY_RE.match(cleaned)
    if not m:
        nums = re.findall(r"\b([01])\b.*?(\d{1,3})\b", cleaned)
        if nums:
            label, conf = int(nums[0][0]), _clamp_confidence(int(nums[0][1]))
            return label, conf, None, None
        return None, None, None, f"unparseable: {cleaned[:120]!r}"
    label, conf = int(m.group(1)), _clamp_confidence(int(m.group(2)))
    return label, conf, None, None


def parse_with_reasoning(text: str) -> tuple[int | None, int | None, str | None, str | None]:
    """Parse JSON or fallback 'label confidence' + trailing reasoning."""
    if not text:
        return None, None, None, "empty response"

    stripped = text.strip()

    # Primary: JSON object
    json_match = JSON_RE.search(stripped)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            label = int(obj.get("label"))
            conf = _clamp_confidence(int(obj.get("confidence")))
            reasoning = str(obj.get("reasoning", "")).strip() or None
            if label in (0, 1):
                return label, conf, reasoning, None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: first line label/confidence, remainder = reasoning
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return None, None, None, "empty response"

    label, conf, _, err = parse_label_only(lines[0])
    if not err:
        reasoning = " ".join(lines[1:]).strip() or None
        return label, conf, reasoning, None

    # Prose fallback (common for local LLMs e.g. CodeLlama)
    prose = _parse_prose_label(stripped)
    if prose is not None:
        return prose, 75, stripped[:500], None

    return None, None, None, err


def _parse_prose_label(text: str) -> int | None:
    """Extract binary label from free-text model output."""
    lower = text.lower()
    if re.search(r"\bnon[- ]?bug", lower) or "not a bug" in lower:
        return 0
    if re.search(r"\bbug[- ]?fix", lower) or re.search(r"\bthis is a bug\b", lower):
        return 1
    if re.search(r'"label"\s*:\s*0', lower):
        return 0
    if re.search(r'"label"\s*:\s*1', lower):
        return 1
    return None


def parse_model_output(text: str, *, with_reasoning: bool = False) -> tuple[int | None, int | None, str | None, str | None]:
    if with_reasoning:
        return parse_with_reasoning(text)
    return parse_label_only(text)
