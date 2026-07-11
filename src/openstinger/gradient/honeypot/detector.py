"""
HoneypotDetector — pattern matching engine for GradientHoneypot (v0.9).

check() performs a case-insensitive substring scan of the query against all
registered patterns. This is intentionally O(n_patterns * len(query)) —
patterns should stay under ~40 entries so the scan completes in microseconds.

Returns HoneypotAlert on first match, None if no pattern fires.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openstinger.gradient.honeypot.models import HoneypotAlert
from openstinger.gradient.honeypot.registry import HoneypotRegistry

if TYPE_CHECKING:
    from openstinger.config import HarnessConfig

logger = logging.getLogger(__name__)


class HoneypotDetector:
    """
    Holds the pattern registry and exposes the check() method.

    Instantiated once at Gradient server startup alongside DriftDetector.

    Usage:
        detector = HoneypotDetector(cfg)
        alert = detector.check(query, tool="memory_search", agent_namespace="main")
        if alert:
            await _handle_honeypot_alert(alert)
    """

    def __init__(self, cfg: "HarnessConfig") -> None:
        self._registry = HoneypotRegistry(cfg)
        self._enabled: bool = True

        honeypot_cfg = getattr(cfg.gradient, "honeypot", None)
        if honeypot_cfg is not None:
            self._enabled = bool(honeypot_cfg.enabled)

        logger.info("HoneypotDetector initialised (enabled=%s)", self._enabled)

    def check(
        self,
        query:           str,
        tool:            str,
        agent_namespace: str,
    ) -> HoneypotAlert | None:
        """
        Scan query for honeypot patterns. Returns HoneypotAlert on first match, else None.

        Args:
            query:           Raw query string from the MCP tool call.
            tool:            Tool name ('memory_search', 'memory_get_entity', 'knowledge_search').
            agent_namespace: Agent namespace for context in the alert.

        Returns:
            HoneypotAlert if a match found, None otherwise.
        """
        if not self._enabled or not query:
            return None

        query_lower = query.lower()

        for pattern, source in self._registry.all_patterns():
            if pattern.lower() in query_lower:
                alert = HoneypotAlert(
                    agent_namespace = agent_namespace,
                    tool_called     = tool,
                    query_text      = query,
                    matched_pattern = pattern,
                    pattern_source  = source,
                )
                logger.error(
                    "HONEYPOT TRIGGERED: namespace=%s tool=%s pattern=%r (source=%s) query=%r",
                    agent_namespace, tool, pattern, source, query[:120],
                )
                return alert

        return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def pattern_count(self) -> dict[str, int]:
        return self._registry.total_count

    @property
    def default_patterns(self) -> list[str]:
        return list(self._registry.default_patterns)

    @property
    def custom_patterns(self) -> list[str]:
        return list(self._registry.custom_patterns)
