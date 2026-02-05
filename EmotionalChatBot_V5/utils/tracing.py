"""
Tracing switch utilities (LangSmith / LangChain tracing).

Goal:
- Allow dynamically enabling/disabling LangSmith tracing *without* deleting API keys.
- Provide a safe decorator that becomes a no-op when disabled.

How to use:
- Set env var `LTSR_LANGSMITH_ENABLED`:
  - "1" / "true" / "yes" / "on"  -> enabled
  - "0" / "false" / "no" / "off" -> disabled

Notes:
- Even if LANGCHAIN_TRACING_V2 is true, we can still force-disable tracing via this flag.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_langsmith_enabled(default: bool = True) -> bool:
    """
    Decide whether LangSmith tracing should be enabled for this process.

    Precedence:
    1) LTSR_LANGSMITH_ENABLED (project-level switch)
    2) default
    """
    raw = os.getenv("LTSR_LANGSMITH_ENABLED")
    if raw is None:
        return default
    return _truthy(raw)


def trace_if_enabled(
    *,
    name: str,
    run_type: str = "chain",
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator factory.
    - If tracing disabled -> returns identity decorator.
    - If tracing enabled  -> returns langsmith.traceable(...) decorator.
    """

    def _identity(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    if not is_langsmith_enabled(default=True):
        return _identity

    try:
        from langsmith import traceable  # type: ignore
    except Exception:
        return _identity

    return traceable(run_type=run_type, name=name, tags=tags, metadata=metadata)

