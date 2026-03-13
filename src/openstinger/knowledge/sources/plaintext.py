"""Plain text source extractor — passthrough."""

from __future__ import annotations


async def extract(text: str) -> str:
    """Return the text as-is after basic normalisation."""
    return text.strip()
