"""CodeLlama-13b-Instruct local backend with prompt-based tool calling."""

from __future__ import annotations

import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_TOKENS_PER_STEP, TEMPERATURE
from llm_types import ModelResponse, ToolCall

CODELLAMA_PATH = os.environ.get(
    "CODELLAMA_PATH",
    "/gpuhome/shanto1/models/CodeLlama-13b-Instruct-hf",
)

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"Loading CodeLlama-13b-Instruct from {CODELLAMA_PATH} ...")
    _tokenizer = AutoTokenizer.from_pretrained(CODELLAMA_PATH)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        CODELLAMA_PATH,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    _model.eval()
    print("  CodeLlama loaded.")
    return _model, _tokenizer


def reset_model():
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    _tokenizer = None
    torch.cuda.empty_cache()


def _build_tools_prompt(tool_schemas: list[dict]) -> str:
    lines = [
        "You have these tools. To call a tool, output EXACTLY one block per tool:",
        '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>',
        "",
        "CRITICAL RULES:",
        "- Do NOT describe tool calls in prose. Output the <tool_call> XML block.",
        "- For submit_answer, label MUST be integer 0 (Non-Bug-Fix) or 1 (Bug-Fix).",
        "- Example: <tool_call>{\"name\": \"submit_answer\", \"arguments\": {\"label\": 1, \"confidence\": 85, \"reasoning\": \"...\", \"backward_reasoning\": \"...\"}}</tool_call>",
        "",
        "Available tools:",
    ]
    for schema in tool_schemas:
        lines.append(f"- {schema['name']}: {schema['description']}")
    lines.append("")
    lines.append("You MUST call submit_answer to finish.")
    return "\n".join(lines)


def _build_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    system_msg = ""
    if tools:
        tool_prompt = _build_tools_prompt(tools)
    else:
        tool_prompt = ""

    parts = []
    pending_user = ""
    first_user = True

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            system_msg = content + ("\n\n" + tool_prompt if tool_prompt else "")
        elif role == "user":
            pending_user = content
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            block = f"[Tool Result from {tool_name}]:\n{content}"
            pending_user = f"{pending_user}\n\n{block}" if pending_user else block
        elif role == "assistant":
            if first_user:
                parts.append(
                    f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{pending_user} [/INST]"
                )
                first_user = False
            else:
                parts.append(f"[INST] {pending_user} [/INST]")
            assistant_content = content or ""
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc)
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    assistant_content += (
                        f'\n<tool_call>{json.dumps({"name": func["name"], "arguments": args})}</tool_call>'
                    )
            parts.append(f" {assistant_content} </s>")
            pending_user = ""

    if pending_user or first_user:
        if first_user:
            parts.append(
                f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{pending_user} [/INST]"
            )
        else:
            parts.append(f"[INST] {pending_user} [/INST]")

    return "<s>" + "".join(parts)


def _parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    tool_calls = []
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            tool_calls.append(ToolCall(
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
            ))
        except json.JSONDecodeError:
            continue

    # Fallback: prose submit_answer(1, 85, "reasoning...")
    if not tool_calls:
        prose = re.search(
            r"submit_answer\s*\(\s*([01])\s*,\s*(\d+)\s*,\s*[\"'](.+?)[\"']\s*\)",
            text,
            re.DOTALL,
        )
        if prose:
            tool_calls.append(ToolCall(
                name="submit_answer",
                arguments={
                    "label": int(prose.group(1)),
                    "confidence": int(prose.group(2)),
                    "reasoning": prose.group(3),
                },
            ))

    remaining = pattern.sub("", text).strip()
    return remaining, tool_calls


def generate(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_new_tokens: int = MAX_TOKENS_PER_STEP,
    temperature: float = TEMPERATURE,
    enable_thinking: bool = False,
) -> ModelResponse:
    model, tokenizer = _load_model()
    prompt = _build_prompt(messages, tools)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": min(max_new_tokens, 1024),
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature <= 0.01:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    remaining, tool_calls = _parse_tool_calls(raw_text)

    return ModelResponse(
        text=remaining,
        tool_calls=tool_calls,
        raw_output=raw_text,
    )


def format_tool_result(tool_name: str, result: str) -> dict:
    return {"role": "tool", "content": result, "name": tool_name}


def format_assistant_with_tool_calls(
    text: str,
    tool_calls: list[ToolCall],
    call_ids: list[str] | None = None,
) -> dict:
    content = text if text else ""
    if tool_calls:
        blocks = [
            f'<tool_call>{json.dumps({"name": tc.name, "arguments": tc.arguments})}</tool_call>'
            for tc in tool_calls
        ]
        content = content + "\n" + "\n".join(blocks) if content else "\n".join(blocks)
    return {"role": "assistant", "content": content}
