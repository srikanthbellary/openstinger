"""
Agent namespace lifecycle management.

Each named agent gets:
  - A private temporal graph: openstinger_temporal_<agent_id[:8]>
  - Shared knowledge graph: openstinger_knowledge (read + write)
  - A row in agent_registry SQLite table

Anonymous/task agents share the default temporal graph but get read-only
access via AnonymousAgentContext.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentRecord:
    """Represents a registered agent namespace."""
    agent_id: str
    agent_name: str
    temporal_graph: str
    status: str = "active"
    created_at: int = 0
    last_active: int = 0
    config_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "temporal_graph": self.temporal_graph,
            "status": self.status,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


async def create_namespace(
    name: str,
    db: Any,
    driver: Any,
    config_hash: Optional[str] = None,
) -> AgentRecord:
    """
    Create a new agent namespace with its own temporal graph.

    Args:
        name:        Human-readable agent name (e.g. "research-agent").
        db:          OperationalDBAdapter instance.
        driver:      FalkorDBDriver instance.
        config_hash: Optional SHA-256 of the agent's HarnessConfig.

    Returns:
        AgentRecord with the new agent's id and graph name.
    """
    agent_id = str(uuid.uuid4())
    graph_name = f"openstinger_temporal_{agent_id[:8]}"
    now = int(time.time())

    # Initialize temporal graph schema for new namespace
    try:
        from openstinger.temporal.falkordb_driver import TEMPORAL_SCHEMA_QUERIES
        original_graph = driver.temporal_graph_name
        driver.temporal_graph_name = graph_name
        driver._temporal = driver._client.select_graph(graph_name)
        await driver._init_graph_schema(driver._temporal, TEMPORAL_SCHEMA_QUERIES, graph_name)
        # Restore default
        driver.temporal_graph_name = original_graph
        driver._temporal = driver._client.select_graph(original_graph)
    except Exception as exc:
        logger.warning("Failed to init temporal schema for %s: %s", graph_name, exc)

    # Register in DB
    try:
        await db.create_agent_registry_row(
            agent_id=agent_id,
            agent_name=name,
            temporal_graph=graph_name,
            config_hash=config_hash,
            created_at=now,
        )
    except Exception as exc:
        # agent_registry table may not exist if DB hasn't been migrated
        logger.warning("Could not persist agent registry row: %s", exc)

    record = AgentRecord(
        agent_id=agent_id,
        agent_name=name,
        temporal_graph=graph_name,
        created_at=now,
        last_active=now,
        config_hash=config_hash,
    )
    logger.info("Created agent namespace: name=%s id=%s graph=%s", name, agent_id[:8], graph_name)
    return record


async def archive_namespace(
    agent_id: str,
    db: Any,
) -> bool:
    """
    Mark an agent namespace as archived (soft-delete).

    The temporal graph data is preserved. The agent can no longer receive
    new ingestion but its memory remains queryable.

    Returns True if the namespace was found and archived.
    """
    try:
        await db.archive_agent_registry_row(agent_id=agent_id)
        logger.info("Archived agent namespace: %s", agent_id[:8])
        return True
    except Exception as exc:
        logger.warning("Failed to archive namespace %s: %s", agent_id[:8], exc)
        return False


async def list_namespaces(
    db: Any,
    include_archived: bool = False,
) -> list[AgentRecord]:
    """
    List all registered agent namespaces.

    Falls back to an empty list if the agent_registry table doesn't exist.
    """
    try:
        rows = await db.list_agent_registry_rows(include_archived=include_archived)
        return [
            AgentRecord(
                agent_id=r.agent_id,
                agent_name=r.agent_name,
                temporal_graph=r.temporal_graph,
                status=r.status,
                created_at=r.created_at or 0,
                last_active=r.last_active or 0,
                config_hash=r.config_hash,
            )
            for r in rows
        ]
    except Exception as exc:
        logger.debug("list_namespaces: agent_registry unavailable: %s", exc)
        return []


def config_hash(config_dict: dict) -> str:
    """Compute a stable SHA-256 hash of a config dict."""
    canonical = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
