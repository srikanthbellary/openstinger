"""
OpenAI embedding client for the temporal engine.

Adapted from graphiti-core v0.24.0 openai_embedder.py:
  - Import paths updated
  - Batch embedding support (multiple texts in one API call)
  - Retry logic via tenacity
  - Returns list[float] per text
"""

from __future__ import annotations

import asyncio
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


class OpenAIEmbedder:
    """Async-compatible OpenAI-compatible embeddings wrapper.

    Works with any OpenAI-compatible API (OpenAI, Novita, etc.)
    Set base_url to point at an alternative provider.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a float vector."""
        results = await self.embed_batch([text])
        return results[0]

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call. Returns list of float vectors."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            ),
        )

        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]
