"""
Retry decorator with exponential backoff + jitter — v0.9 reliability feature.

Applied to LLM call sites (classification, evolve, correction, evaluation).
Do NOT apply to FalkorDB or embedding calls — those are handled by CircuitBreaker.

Delay formula: min(base_delay * 2^attempt, max_delay) [+ uniform(0, 1) if jitter=True]
"""

import asyncio
import functools
import logging
import random
from typing import Callable

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts:         int   = 3,
    base_delay:           float = 1.0,
    max_delay:            float = 30.0,
    jitter:               bool  = True,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for async functions. Retries on retryable_exceptions using
    exponential backoff with optional jitter.

    Args:
        max_attempts:         Total attempts (including first try).
        base_delay:           Base delay in seconds.
        max_delay:            Cap on computed delay in seconds.
        jitter:               Add random uniform(0, 1) seconds to each delay.
        retryable_exceptions: Exception types that trigger a retry.

    Usage:
        @with_retry(max_attempts=cfg.resilience.retry_max_attempts,
                    base_delay=cfg.resilience.retry_base_delay,
                    max_delay=cfg.resilience.retry_max_delay,
                    jitter=cfg.resilience.retry_jitter)
        async def my_llm_call(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay += random.uniform(0, 1)
                    logger.warning(
                        "[retry] %s attempt %d/%d failed: %s. Retrying in %.2fs",
                        fn.__name__, attempt + 1, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
