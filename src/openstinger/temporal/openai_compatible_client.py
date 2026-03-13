"""
OpenAI-compatible LLM client for the temporal engine.

Drop-in replacement for AnthropicClient when using OpenAI-compatible providers
(Novita, DeepSeek, etc.). Implements the same interface: complete(), complete_json(),
complete_with_tools().

Tool format conversion: Anthropic uses `input_schema`, OpenAI uses `parameters`.
This client handles the conversion transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """
    Async-compatible wrapper around any OpenAI-compatible chat API.

    Implements the same interface as AnthropicClient so it can be used
    as a drop-in replacement throughout the codebase.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek/deepseek-v3.2",
        fast_model: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.fast_model = fast_model or model
        self.max_tokens = max_tokens
        client_kwargs: dict = {"api_key": api_key or "placeholder"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
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
        """Simple text completion. Returns the text content of the first message."""
        model = self.fast_model if use_fast_model else self.model
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
        )
        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
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
        """Completion expecting JSON. Parses and returns the dict."""
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
        Completion using tool_use for structured output.

        Accepts Anthropic-format tool defs (with `input_schema`) and converts
        them to OpenAI format (with `parameters`) automatically.
        Returns the first tool call's arguments as a dict.
        """
        model = self.fast_model if use_fast_model else self.model
        loop = asyncio.get_event_loop()

        # Convert Anthropic tool format → OpenAI tool format
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    # Anthropic uses input_schema, OpenAI uses parameters
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })

        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=openai_tools,
                tool_choice="required",
            ),
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments)

        raise ValueError("LLM response contained no tool_use block")
