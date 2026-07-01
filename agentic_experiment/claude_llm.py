"""Anthropic Claude API backend with native tool calling."""

from __future__ import annotations

import json
import os

import anthropic

from config import MAX_TOKENS_PER_STEP, TEMPERATURE
from llm_types import ModelResponse, ToolCall

_client: anthropic.Anthropic | None = None
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        _client = anthropic.Anthropic(api_key=api_key)
        print(f"  Claude model: {CLAUDE_MODEL}")
    return _client


def _load_model():
    _get_client()
    print("  Anthropic client ready.")
    return None, None


def reset_model():
    global _client
    _client = None


def _build_tools_for_api(tool_schemas: list[dict]) -> list[dict]:
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["input_schema"],
        }
        for s in tool_schemas
    ]


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_prompt = ""
    api_messages = []
    last_tool_use_ids: dict[str, list[str]] = {}

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user":
            api_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if msg.get("tool_calls"):
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                call_ids = msg.get("_call_ids", [])
                last_tool_use_ids = {}
                for i, tc in enumerate(msg["tool_calls"]):
                    func = tc.get("function", tc)
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_id = call_ids[i] if i < len(call_ids) else f"toolu_gen_{i:04d}"
                    tool_name = func["name"]
                    blocks.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": args,
                    })
                    last_tool_use_ids.setdefault(tool_name, []).append(tool_id)
                api_messages.append({"role": "assistant", "content": blocks})
            else:
                api_messages.append({"role": "assistant", "content": content})
                last_tool_use_ids = {}
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            tool_use_id = "toolu_unknown"
            if last_tool_use_ids.get(tool_name):
                tool_use_id = last_tool_use_ids[tool_name].pop(0)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
            if api_messages and api_messages[-1]["role"] == "user":
                last = api_messages[-1]["content"]
                if isinstance(last, list):
                    last.append(tool_result_block)
                else:
                    api_messages[-1]["content"] = [
                        {"type": "text", "text": last},
                        tool_result_block,
                    ]
            else:
                api_messages.append({"role": "user", "content": [tool_result_block]})

    return system_prompt, api_messages


def generate(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int = MAX_TOKENS_PER_STEP,
    temperature: float = TEMPERATURE,
    enable_thinking: bool = False,
) -> ModelResponse:
    client = _get_client()
    system_prompt, api_messages = _convert_messages(messages)
    api_tools = _build_tools_for_api(tools) if tools else []

    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_new_tokens,
        "messages": api_messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if api_tools:
        kwargs["tools"] = api_tools
    if temperature > 0.01:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)

    text_parts = []
    tool_calls = []
    call_ids = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(name=block.name, arguments=block.input))
            call_ids.append(block.id)

    result = ModelResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        raw_output=str(response.content),
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )
    result._call_ids = call_ids
    return result


def format_tool_result(tool_name: str, result: str) -> dict:
    return {"role": "tool", "content": result, "name": tool_name}


def format_assistant_with_tool_calls(
    text: str,
    tool_calls: list[ToolCall],
    call_ids: list[str] | None = None,
) -> dict:
    msg = {"role": "assistant", "content": text if text else ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                }
            }
            for tc in tool_calls
        ]
    if call_ids:
        msg["_call_ids"] = call_ids
    return msg
