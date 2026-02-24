"""
Graph node models for the temporal engine.

Adapted from graphiti-core v0.24.0 nodes.py:
  - Import paths updated: graphiti_core.* → openstinger.temporal.*
  - agent_namespace field added to all nodes
  - FalkorDB-compatible serialization (unix timestamps, no multi-label)
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
# Episode node
# ---------------------------------------------------------------------------

class EpisodeNode(BaseModel):
    """
    A raw interaction unit — a turn of conversation, a tool call result,
    a document chunk — stored verbatim.
    """

    uuid: str = Field(default_factory=_new_uuid)
    agent_namespace: str = "default"
    content: str
    source: str = "conversation"          # conversation | tool_result | document | manual
    source_description: str = ""
    created_at: int = Field(default_factory=_now_unix)   # unix timestamp
    valid_at: int = Field(default_factory=_now_unix)      # when it occurred in the world

    model_config = ConfigDict(populate_by_name=True)

    def to_cypher_props(self) -> dict:
        return {
            "uuid": self.uuid,
            "agent_namespace": self.agent_namespace,
            "content": self.content,
            "source": self.source,
            "source_description": self.source_description,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
        }

    @classmethod
    def from_cypher_props(cls, props: dict) -> "EpisodeNode":
        return cls(**props)


# ---------------------------------------------------------------------------
# Entity node
# ---------------------------------------------------------------------------

class EntityNode(BaseModel):
    """
    A deduplicated real-world entity: person, organisation, concept, etc.
    """

    uuid: str = Field(default_factory=_new_uuid)
    agent_namespace: str = "default"
    name: str
    name_normalized: str = ""            # lowercase, stripped for matching
    entity_type: str = "ENTITY"          # PERSON | ORG | CONCEPT | LOCATION | EVENT | ENTITY
    summary: str = ""
    name_embedding: Optional[list[float]] = None
    created_at: int = Field(default_factory=_now_unix)
    episode_count: int = 0

    model_config = ConfigDict(populate_by_name=True)

    def to_cypher_props(self) -> dict:
        props: dict = {
            "uuid": self.uuid,
            "agent_namespace": self.agent_namespace,
            "name": self.name,
            "name_normalized": self.name_normalized or self.name.lower().strip(),
            "entity_type": self.entity_type,
            "summary": self.summary,
            "created_at": self.created_at,
            "episode_count": self.episode_count,
        }
        if self.name_embedding is not None:
            props["name_embedding"] = self.name_embedding
        return props

    @classmethod
    def from_cypher_props(cls, props: dict) -> "EntityNode":
        return cls(**{k: v for k, v in props.items() if k in cls.model_fields})
