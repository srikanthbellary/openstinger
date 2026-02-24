"""
Anthropic LLM client wrapper for the temporal engine.

Adapted from graphiti-core v0.24.0 anthropic_client.py:
  - Import paths updated
  - Retry logic via tenacity
  - Structured output via response_format / tool_use pattern
  - Separate fast_model support (haiku for Tier 3 evaluations)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Thin async-compatible wrapper around the Anthropic messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        fast_model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.fast_model = fast_model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete(
        self,
        system: str,
        user: str,
        use_fast_model: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """
        Simple text completion. Returns the text content of the first message.
        """
        import asyncio

        model = self.fast_model if use_fast_model else self.model
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
            ),
        )
        return response.content[0].text  # type: ignore[index]

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete_json(
        self,
        system: str,
        user: str,
        use_fast_model: bool = False,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Completion expecting a JSON response. Parses and returns the dict.
        Raises ValueError if response is not valid JSON.
        """
        text = await self.complete(system, user, use_fast_model, temperature)
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON: %s", text[:200])
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    async def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        use_fast_model: bool = False,
    ) -> dict[str, Any]:
        """
        Completion using Anthropic tool_use for structured output.
        Returns the tool_use input dict from the first tool call.
        """
        import asyncio

        model = self.fast_model if use_fast_model else self.model
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "any"},
            ),
        )

        for block in response.content:
            if block.type == "tool_use":
                return block.input  # type: ignore[return-value]

        raise ValueError("LLM response contained no tool_use block")
