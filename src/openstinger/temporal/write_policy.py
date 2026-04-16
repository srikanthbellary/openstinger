"""
Memory Write Policy — v0.9 Feature 6.

Intercepts memory write operations before they reach FalkorDB.
Classifies each write intent as ADD / UPDATE / DELETE / NOOP.

NOOP is returned when a near-duplicate already exists in the graph
(detected via vector similarity on TemporalEngine.find_similar_episodes()).

Fails open: any internal error returns ADD to avoid blocking legitimate writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openstinger.temporal.engine import TemporalEngine

logger = logging.getLogger(__name__)


class WriteOp(Enum):
    ADD    = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP   = "NOOP"


@dataclass
class WritePolicyDecision:
    op:            WriteOp
    reason:        str
    existing_uuid: str | None = None


class MemoryWritePolicy:
    """
    Write-side interceptor for memory operations.

    NOOP logic:
      - Only fires on op_type="add" (update/delete are always respected explicitly)
      - Calls engine.find_similar_episodes() with dedup_threshold
      - If a match is found with score >= threshold: returns NOOP with the existing_uuid
      - If the similarity check itself fails: fails open (returns ADD)

    Config:
      dedup_threshold: float — cosine similarity threshold (default: 0.95)
      A very high threshold avoids false-positive NOOPs. Lower it (e.g. 0.85) for
      more aggressive deduplication in high-volume pipelines.
    """

    def __init__(self, dedup_threshold: float = 0.95) -> None:
        self.dedup_threshold = dedup_threshold

    async def evaluate(
        self,
        op_type:         Literal["add", "update", "delete"],
        content:         str,
        agent_namespace: str,
        engine:          "TemporalEngine",
    ) -> WritePolicyDecision:
        """
        Evaluate the write operation and return a policy decision.

        Args:
            op_type:         "add" | "update" | "delete"
            content:         Episode content string
            agent_namespace: Agent namespace for scope isolation
            engine:          TemporalEngine instance (provides find_similar_episodes)

        Returns:
            WritePolicyDecision with op, reason, and (for NOOP) existing_uuid.
        """
        if op_type == "delete":
            return WritePolicyDecision(
                op=WriteOp.DELETE, reason="Explicit delete requested"
            )

        if op_type == "update":
            return WritePolicyDecision(
                op=WriteOp.UPDATE, reason="Explicit update requested"
            )

        # op_type == "add" — check for near-duplicate
        try:
            similar = await engine.find_similar_episodes(
                content         = content,
                agent_namespace = agent_namespace,
                threshold       = self.dedup_threshold,
                limit           = 1,
            )
        except Exception as exc:
            logger.warning(
                "MemoryWritePolicy similarity check failed: %s — defaulting to ADD", exc
            )
            return WritePolicyDecision(
                op=WriteOp.ADD,
                reason="Similarity check failed — defaulting to ADD",
            )

        if similar and similar[0].get("score", 0) >= self.dedup_threshold:
            return WritePolicyDecision(
                op            = WriteOp.NOOP,
                reason        = f"Near-duplicate detected (score={similar[0]['score']:.3f})",
                existing_uuid = similar[0].get("uuid"),
            )

        return WritePolicyDecision(
            op=WriteOp.ADD,
            reason="New episode — no near-duplicate found",
        )
