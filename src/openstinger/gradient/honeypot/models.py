"""
GradientHoneypot models — v0.9.

HoneypotAlert: immutable dataclass produced by HoneypotDetector.check().
Passed to _handle_honeypot_alert() in the Gradient server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class HoneypotAlert:
    """
    Produced when a retrieval tool query matches a honeypot pattern.

    Attributes:
        agent_namespace: Agent namespace that issued the query.
        tool_called:     Which retrieval tool fired (memory_search, memory_get_entity, knowledge_search).
        query_text:      The raw query string that matched.
        matched_pattern: The specific pattern that fired.
        pattern_source:  'default' | 'custom'.
        triggered_at:    UTC datetime of the match.
    """
    agent_namespace: str
    tool_called:     str
    query_text:      str
    matched_pattern: str
    pattern_source:  str   # 'default' | 'custom'
    triggered_at:    datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Use object.__setattr__ since the dataclass is frozen
        if self.triggered_at is None:
            object.__setattr__(self, "triggered_at", datetime.now(timezone.utc))
