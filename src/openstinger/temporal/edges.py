"""
Graph edge models for the temporal engine.

Adapted from graphiti-core v0.24.0 edges.py:
  - Import paths updated
  - agent_namespace field added
  - Bi-temporal fields clarified: valid_from/valid_to (world time),
    recorded_at/expired_at (agent knowledge time)
  - FalkorDB serialization (unix timestamps as integers)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


def _now_unix() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# EntityEdge — the bi-temporal fact edge
# ---------------------------------------------------------------------------

class EntityEdge(BaseModel):
    """
    A directed relationship between two EntityNodes carrying a fact.

    Bi-temporal fields:
      valid_from   / valid_to   — when this fact was/is true in the world
      recorded_at  / expired_at — when the agent learned / superseded this fact

    'expired_at' is NULL while the edge is current.
    Edges are NEVER deleted — only expired.
    """

    uuid: str = Field(default_factory=_new_uuid)
    agent_namespace: str = "default"

    # Endpoints (UUIDs of EntityNodes)
    source_node_uuid: str
    target_node_uuid: str

    # Relation type label (e.g. WORKS_AT, KNOWS, LOCATED_IN)
    relation_type: str

    # The fact statement
    fact: str
    fact_embedding: Optional[list[float]] = None

    # Bi-temporal fields (unix timestamps)
    valid_from: int = Field(default_factory=_now_unix)
    valid_to: Optional[int] = None        # None = open-ended (still true)
    recorded_at: int = Field(default_factory=_now_unix)
    expired_at: Optional[int] = None      # None = still current

    # Metadata
    episodes: list[str] = Field(default_factory=list)   # episode UUIDs that support this fact
    confidence: float = 1.0
    created_at: int = Field(default_factory=_now_unix)

    model_config = ConfigDict(populate_by_name=True)

    @property
    def is_current(self) -> bool:
        return self.expired_at is None

    def expire(self, expired_at: int | None = None) -> None:
        """Mark this edge as superseded."""
        self.expired_at = expired_at or _now_unix()

    def to_cypher_props(self) -> dict:
        props: dict = {
            "uuid": self.uuid,
            "agent_namespace": self.agent_namespace,
            "relation_type": self.relation_type,
            "fact": self.fact,
            "valid_from": self.valid_from,
            "recorded_at": self.recorded_at,
            "episodes": self.episodes,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }
        if self.valid_to is not None:
            props["valid_to"] = self.valid_to
        if self.expired_at is not None:
            props["expired_at"] = self.expired_at
        if self.fact_embedding is not None:
            props["fact_embedding"] = self.fact_embedding
        return props

    @classmethod
    def from_cypher_props(cls, props: dict) -> "EntityEdge":
        return cls(**{k: v for k, v in props.items() if k in cls.model_fields})


# ---------------------------------------------------------------------------
# EpisodicEdge — links an Episode to the Entities it mentions
# ---------------------------------------------------------------------------

class EpisodicEdge(BaseModel):
    """Links an EpisodeNode to the EntityNodes it mentions."""

    uuid: str = Field(default_factory=_new_uuid)
    agent_namespace: str = "default"
    episode_uuid: str
    entity_uuid: str
    created_at: int = Field(default_factory=_now_unix)

    def to_cypher_props(self) -> dict:
        return {
            "uuid": self.uuid,
            "agent_namespace": self.agent_namespace,
            "episode_uuid": self.episode_uuid,
            "entity_uuid": self.entity_uuid,
            "created_at": self.created_at,
        }
