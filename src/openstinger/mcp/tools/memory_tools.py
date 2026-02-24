"""
Tier 1 MCP tool handlers — 9 tools.

Tools:
  1. memory_add            — add an episode manually
  2. memory_query          — hybrid search (BM25 + vector)
  3. memory_search         — BM25-only keyword search
  4. memory_get_entity     — fetch entity by UUID
  5. memory_get_episode    — fetch episode by UUID
  6. memory_job_status     — check ingestion job status
  7. memory_ingest_now     — trigger immediate session ingestion
  8. memory_namespace_status — namespace health + stats (v0.3)
  9. memory_list_agents    — list registered agent namespaces (v0.3)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: memory_add
# ---------------------------------------------------------------------------

async def memory_add(
    engine: Any,
    db: Any,
    content: str,
    source: str = "manual",
    source_description: str = "",
    valid_at_unix: int | None = None,
    agent_namespace: str | None = None,
) -> dict:
    """
    Manually add an episode to the temporal memory graph.

    Returns episode UUID and extraction summary.
    """
    namespace = agent_namespace or engine.agent_namespace
    episode = await engine.add_episode(
        content=content,
        source=source,
        source_description=source_description,
        valid_at=valid_at_unix or int(time.time()),
        agent_namespace=namespace,
    )
    return {
        "success": True,
        "episode_uuid": episode.uuid,
        "namespace": namespace,
    }


# ---------------------------------------------------------------------------
# Tool 2: memory_query
# ---------------------------------------------------------------------------

async def memory_query(
    engine: Any,
    query: str,
    limit: int = 10,
    include_expired: bool = False,
    agent_namespace: str | None = None,
) -> dict:
    """
    Hybrid search: BM25 keyword + vector similarity on episodes, entities, and facts.
    """
    namespace = agent_namespace or engine.agent_namespace
    results = await engine.query_memory(
        query=query,
        agent_namespace=namespace,
        limit=limit,
        include_expired=include_expired,
    )
    return {
        "query": query,
        "namespace": namespace,
        "episodes": results.get("episodes", []),
        "entities": results.get("entities", []),
        "facts": results.get("facts", []),
    }


# ---------------------------------------------------------------------------
# Tool 3: memory_search
# ---------------------------------------------------------------------------

async def memory_search(
    engine: Any,
    query: str,
    search_type: str = "all",
    limit: int = 10,
    agent_namespace: str | None = None,
) -> dict:
    """
    BM25 full-text search.
    search_type: 'episodes' | 'entities' | 'facts' | 'all'
    """
    namespace = agent_namespace or engine.agent_namespace
    driver = engine.driver
    results: dict[str, list] = {}

    if search_type in ("episodes", "all"):
        rows = await driver.query_temporal(
            """
            CALL db.idx.fulltext.queryNodes('Episode', $query)
            YIELD node, score
            WHERE node.agent_namespace = $ns
            RETURN node.uuid AS uuid, node.content AS content, score
            ORDER BY score DESC LIMIT $limit
            """,
            {"query": query, "ns": namespace, "limit": limit},
        )
        results["episodes"] = rows

    if search_type in ("entities", "all"):
        rows = await driver.query_temporal(
            """
            CALL db.idx.fulltext.queryNodes('Entity', $query)
            YIELD node, score
            WHERE node.agent_namespace = $ns
            RETURN node.uuid AS uuid, node.name AS name,
                   node.entity_type AS entity_type, score
            ORDER BY score DESC LIMIT $limit
            """,
            {"query": query, "ns": namespace, "limit": limit},
        )
        results["entities"] = rows

    if search_type in ("facts", "all"):
        rows = await driver.query_temporal(
            """
            MATCH ()-[r:RELATES_TO {agent_namespace: $ns}]-()
            WHERE r.fact CONTAINS $query AND r.expired_at IS NULL
            RETURN r.uuid AS uuid, r.fact AS fact,
                   r.relation_type AS relation_type
            LIMIT $limit
            """,
            {"query": query, "ns": namespace, "limit": limit},
        )
        results["facts"] = rows

    return {"query": query, "namespace": namespace, **results}


# ---------------------------------------------------------------------------
# Tool 4: memory_get_entity
# ---------------------------------------------------------------------------

async def memory_get_entity(engine: Any, uuid: str) -> dict:
    """Fetch entity node by UUID including current and expired edges."""
    driver = engine.driver

    entity_rows = await driver.query_temporal(
        "MATCH (e:Entity {uuid: $uuid}) RETURN e",
        {"uuid": uuid},
    )
    if not entity_rows:
        return {"found": False, "uuid": uuid}

    # Fetch current edges
    current_edges = await driver.query_temporal(
        """
        MATCH (e:Entity {uuid: $uuid})-[r:RELATES_TO]->(tgt:Entity)
        WHERE r.expired_at IS NULL
        RETURN r.uuid AS edge_uuid, r.fact AS fact, r.relation_type AS relation_type,
               tgt.name AS target_name, r.valid_from AS valid_from
        ORDER BY r.valid_from DESC LIMIT 20
        """,
        {"uuid": uuid},
    )

    return {
        "found": True,
        "entity": entity_rows[0],
        "current_edges": current_edges,
    }


# ---------------------------------------------------------------------------
# Tool 5: memory_get_episode
# ---------------------------------------------------------------------------

async def memory_get_episode(engine: Any, uuid: str) -> dict:
    """Fetch episode node by UUID."""
    row = await engine.get_episode(uuid)
    if not row:
        return {"found": False, "uuid": uuid}
    return {"found": True, "episode": row}


# ---------------------------------------------------------------------------
# Tool 6: memory_job_status
# ---------------------------------------------------------------------------

async def memory_job_status(db: Any, job_uuid: str) -> dict:
    """Check the status of an ingestion job."""
    job = await db.get_job(job_uuid)
    if not job:
        return {"found": False, "job_uuid": job_uuid}
    return {
        "found": True,
        "job_uuid": job_uuid,
        "status": job.status,
        "episodes_processed": job.episodes_processed,
        "entities_created": job.entities_created,
        "edges_created": job.edges_created,
        "edges_expired": job.edges_expired,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


# ---------------------------------------------------------------------------
# Tool 7: memory_ingest_now
# ---------------------------------------------------------------------------

async def memory_ingest_now(
    scheduler: Any,
    agent_namespace: str | None = None,
) -> dict:
    """
    Trigger immediate ingestion from session files (bypasses poll interval).
    Returns immediately — ingestion runs in the background.
    Check memory_namespace_status to see progress.
    """
    import asyncio
    namespace = agent_namespace or "default"
    asyncio.create_task(scheduler.ingest_now(namespace))
    return {
        "success": True,
        "namespace": namespace,
        "status": "ingestion started in background — call memory_namespace_status to check progress",
    }


# ---------------------------------------------------------------------------
# Tool 8: memory_namespace_status (v0.3)
# ---------------------------------------------------------------------------

async def memory_namespace_status(
    engine: Any,
    db: Any,
    agent_namespace: str | None = None,
) -> dict:
    """
    Return health and stats for a namespace.
    """
    namespace = agent_namespace or engine.agent_namespace

    # Episode count
    ep_rows = await engine.driver.query_temporal(
        "MATCH (ep:Episode {agent_namespace: $ns}) RETURN count(ep) AS count",
        {"ns": namespace},
    )
    episode_count = ep_rows[0]["count"] if ep_rows else 0

    # Entity count
    ent_rows = await engine.driver.query_temporal(
        "MATCH (e:Entity {agent_namespace: $ns}) RETURN count(e) AS count",
        {"ns": namespace},
    )
    entity_count = ent_rows[0]["count"] if ent_rows else 0

    # Current edge count
    edge_rows = await engine.driver.query_temporal(
        """
        MATCH ()-[r:RELATES_TO {agent_namespace: $ns}]->()
        WHERE r.expired_at IS NULL
        RETURN count(r) AS count
        """,
        {"ns": namespace},
    )
    edge_count = edge_rows[0]["count"] if edge_rows else 0

    # Session state
    state = await db.get_session_state(namespace)

    return {
        "namespace": namespace,
        "episode_count": episode_count,
        "entity_count": entity_count,
        "current_edge_count": edge_count,
        "session_count": state.session_count,
        "registry_size": engine.entity_registry.cache_size(),
    }


# ---------------------------------------------------------------------------
# Tool 9: memory_list_agents (v0.3)
# ---------------------------------------------------------------------------

async def memory_list_agents(scheduler: Any) -> dict:
    """List all registered agent namespaces with their ingestion status."""
    namespaces = scheduler.list_namespaces()
    return {
        "agent_count": len(namespaces),
        "agents": [
            {"namespace": ns, "reader_active": ns in scheduler._readers}
            for ns in namespaces
        ],
    }
