"""
Timeout decorator — v0.9 reliability feature.

Wraps async MCP tool handlers with asyncio.wait_for().
Returns a structured error dict on timeout instead of raising —
prevents the MCP client from hanging on a stuck tool call.
"""

import asyncio
import functools
from typing import Callable


def with_timeout(timeout_seconds: float) -> Callable:
    """
    Decorator that wraps an async function with asyncio.wait_for().

    On timeout: returns {"error": "timeout", ...} — never raises to the caller.
    Apply to MCP tool handlers that touch FalkorDB or LLM endpoints.

    Args:
        timeout_seconds: Max seconds to allow the function to run.

    Usage:
        @with_timeout(cfg.resilience.tool_timeout_seconds)
        async def my_handler(args):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    fn(*args, **kwargs), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                return {
                    "error":   "timeout",
                    "message": (
                        f"Tool '{fn.__name__}' exceeded {timeout_seconds}s timeout. "
                        "The operation was cancelled. Try again or reduce the scope of your query."
                    ),
                    "tool": fn.__name__,
                }
        return wrapper
    return decorator
