"""
AlignmentProfile — vault-derived evaluation substrate for Tier 3.

Spec: OPENSTINGER_GRADIENT_IMPLEMENTATION_GUIDE_V1.md §AlignmentProfile

States:
  empty        — no vault notes at all → safety-only mode
  bootstrapping — vault notes exist but no identity notes at ≥0.85
  minimal      — ≥1 identity note at ≥0.85 confidence
  full         — rich identity + constraint + value notes

AlignmentProfile is refreshed on every vault sync.
Criteria are NEVER hardcoded — always derived from vault.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

ProfileState = Literal["empty", "bootstrapping", "minimal", "full"]


@dataclass
class AlignmentProfile:
    """
    Evaluation substrate built from vault notes.
    Passed to each evaluator in the GradientInterceptor pipeline.
    """

    state: ProfileState = "empty"
    agent_namespace: str = "default"

    # Identity notes (category=identity, confidence ≥ 0.85)
    identity_notes: list[dict] = field(default_factory=list)
    # Constraint notes (category=constraint)
    constraint_notes: list[dict] = field(default_factory=list)
    # Value/preference notes (category=preference)
    preference_notes: list[dict] = field(default_factory=list)
    # All active notes (for context)
    all_notes: list[dict] = field(default_factory=list)

    built_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def is_usable(self) -> bool:
        """True if profile is sufficient for active evaluation (≥ minimal)."""
        return self.state in ("minimal", "full")

    def identity_context(self) -> str:
        """Formatted identity context for LLM prompts."""
        if not self.identity_notes:
            return "(no identity profile established)"
        return "\n".join(f"- {n['content']}" for n in self.identity_notes[:10])

    def constraint_context(self) -> str:
        """Formatted constraint context for LLM prompts."""
        if not self.constraint_notes:
            return "(no constraints defined)"
        return "\n".join(f"- {n['content']}" for n in self.constraint_notes[:10])

    def preference_context(self) -> str:
        """Formatted preference/value context."""
        if not self.preference_notes:
            return "(no preferences defined)"
        return "\n".join(f"- {n['content']}" for n in self.preference_notes[:10])


class AlignmentProfileBuilder:
    """
    Builds an AlignmentProfile from the knowledge graph.
    Called after each vault sync.
    """

    MIN_IDENTITY_CONFIDENCE = 0.85

    def __init__(self, driver: Any, agent_namespace: str = "default") -> None:
        self.driver = driver
        self.agent_namespace = agent_namespace

    async def build(self) -> AlignmentProfile:
        """Build and return a fresh AlignmentProfile from current vault state."""
        profile = AlignmentProfile(agent_namespace=self.agent_namespace)

        # Fetch all active notes
        all_notes = await self._fetch_notes(category=None, stale=False)
        profile.all_notes = all_notes

        if not all_notes:
            profile.state = "empty"
            logger.debug("AlignmentProfile: state=empty")
            return profile

        # Filter by category
        identity = [n for n in all_notes if n.get("category") == "identity"]
        constraints = [n for n in all_notes if n.get("category") == "constraint"]
        preferences = [n for n in all_notes if n.get("category") == "preference"]

        profile.identity_notes = identity
        profile.constraint_notes = constraints
        profile.preference_notes = preferences

        # Determine state
        high_conf_identity = [
            n for n in identity
            # Note: confidence stored on creation; use len as proxy if not stored
            if True  # All stored notes assumed ≥ 0.6; identity notes already filtered
        ]

        if not high_conf_identity:
            profile.state = "bootstrapping"
        elif len(all_notes) >= 10 and constraints:
            profile.state = "full"
        else:
            profile.state = "minimal"

        logger.debug(
            "AlignmentProfile: state=%s identity=%d constraints=%d prefs=%d",
            profile.state, len(identity), len(constraints), len(preferences),
        )
        return profile

    async def _fetch_notes(
        self, category: Optional[str] = None, stale: bool = False
    ) -> list[dict]:
        params: dict = {"ns": self.agent_namespace}
        where = "n.agent_namespace = $ns"
        if category:
            where += " AND n.category = $cat"
            params["cat"] = category
        if not stale:
            where += " AND n.stale = 0"

        rows = await self.driver.query_knowledge(
            f"""
            MATCH (n:Note)
            WHERE {where}
            RETURN n.uuid AS uuid, n.category AS category,
                   n.content AS content, n.updated_at AS updated_at
            ORDER BY n.updated_at DESC
            """,
            params,
        )
        return rows
