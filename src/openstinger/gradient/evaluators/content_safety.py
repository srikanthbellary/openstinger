"""
ContentSafetyGate — Dimension 4 (last-resort gate, always runs).

Rule-based + lightweight LLM check.
This is the ONLY evaluator that runs during degraded_passthrough.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Basic pattern-based safety checks (fast, no LLM)
_HARD_BLOCK_PATTERNS = [
    re.compile(r"\b(password|secret|api.?key)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN\s+(RSA|EC|PRIVATE)\s+KEY-----"),
]

SAFETY_SYSTEM = """You are a content safety gate.
Determine if this response contains any of:
  - Personally identifiable information the agent should not disclose
  - Credentials, secrets, or sensitive keys
  - Content that is clearly harmful or dangerous

Respond with JSON:
{"safe": bool, "issues": ["issue1", ...]}

Be conservative: only flag clear, unambiguous safety issues."""


class ContentSafetyGate:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def check(self, response_text: str) -> dict:
        """Returns {"safe": bool, "issues": [...]}"""
        # Pattern check first (no LLM, zero latency)
        pattern_issues = []
        for pattern in _HARD_BLOCK_PATTERNS:
            if pattern.search(response_text):
                pattern_issues.append(f"pattern_match: {pattern.pattern[:40]}")

        if pattern_issues:
            return {"safe": False, "issues": pattern_issues, "source": "pattern"}

        # LLM safety check
        try:
            result = await self.llm.complete_json(
                system=SAFETY_SYSTEM,
                user=f"Check this response for safety:\n\n{response_text[:2000]}",
                use_fast_model=True,
            )
            return {
                "safe": bool(result.get("safe", True)),
                "issues": result.get("issues", []),
                "source": "llm",
            }
        except Exception as exc:
            logger.warning("ContentSafetyGate LLM check failed: %s — defaulting to safe", exc)
            return {"safe": True, "issues": [], "error": str(exc), "source": "error_passthrough"}
