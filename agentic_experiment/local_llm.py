"""
LLM backend router.

Set LLM_BACKEND=claude (default) or LLM_BACKEND=codellama before calling generate().
"""

from __future__ import annotations

import os

from llm_types import ModelResponse, ToolCall

__all__ = ["ModelResponse", "ToolCall", "generate", "format_tool_result",
           "format_assistant_with_tool_calls", "_load_model", "reset_model",
           "set_backend", "get_backend"]

_BACKEND = os.environ.get("LLM_BACKEND", "claude").lower()


def _backend_module():
    if _BACKEND == "codellama":
        import codellama_llm as mod
    else:
        import claude_llm as mod
    return mod


def set_backend(name: str):
    global _BACKEND
    name = name.lower()
    if name != _BACKEND:
        try:
            _backend_module().reset_model()
        except Exception:
            pass
    _BACKEND = name
    os.environ["LLM_BACKEND"] = _BACKEND


def get_backend() -> str:
    return _BACKEND


def _load_model():
    return _backend_module()._load_model()


def reset_model():
    _backend_module().reset_model()


def generate(*args, **kwargs) -> ModelResponse:
    return _backend_module().generate(*args, **kwargs)


def format_tool_result(tool_name: str, result: str) -> dict:
    return _backend_module().format_tool_result(tool_name, result)


def format_assistant_with_tool_calls(text, tool_calls, call_ids=None) -> dict:
    return _backend_module().format_assistant_with_tool_calls(
        text, tool_calls, call_ids
    )
