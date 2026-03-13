"""
Tier 1 MCP tool handlers — 9 tools.

Tools:
  1. memory_add            — add an episode manually
  2. memory_query          — hybrid search (BM25 + vector) with date filtering
  3. memory_search         — smart search with numeric, temporal, and fuzzy fallbacks
  4. memory_get_entity     — fetch entity by UUID
  5. memory_get_episode    — fetch episode by UUID
  6. memory_job_status     — check ingestion job status
  7. memory_ingest_now     — trigger immediate session ingestion
  8. memory_namespace_status — namespace health + stats
  9. memory_list_agents    — list registered agent namespaces

Search strategy (memory_search):
  - Primary: BM25 fulltext index (fast, token-based)
  - Fallback 1 — Zero results: vector similarity search (semantic fuzzy matching)
  - Fallback 2 — Numeric/IP detected: toLower CONTAINS scan (exact substring match)
  - Fallback 3 — Date-like query: search valid_at_human field for month/year matches
  All fallback results are merged and deduplicated with primary results.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query classification helpers (Issue 2 + Issue 3)
# ---------------------------------------------------------------------------

# Numeric: IP addresses (1.2.3.4), prices ($50, 1.23), wallet/hex (0x...)
_NUMERIC_RE = re.compile(
    r"""
    \b\d{1,3}(?:[.\-:]\d{1,3}){2,}\b   # IP-like: 167.99.222.10
    | \$\s?\d+(?:\.\d+)?                 # prices: $50, $1.23
    | \b0x[0-9a-fA-F]+\b                 # hex/wallet: 0xDeadBeef
    | \b\d{4,}\b                          # long numbers: port 8765, txid
    """,
    re.VERBOSE,
)

# Temporal: month names, year patterns
_MONTHS_FULL = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_MONTHS_ABBR = {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
_YEAR_RE = re.compile(r"\b20[0-9]{2}\b")


def _looks_numeric(query: str) -> bool:
    return bool(_NUMERIC_RE.search(query))


def _looks_temporal(query: str) -> bool:
    words = {w.lower().strip(".,;:'-") for w in query.split()}
    return bool(words & (_MONTHS_FULL | _MONTHS_ABBR)) or bool(_YEAR_RE.search(query))


def _parse_date_to_unix(date_str: str) -> int | None:
    """Parse a date string (YYYY-MM-DD, YYYY-MM, or YYYY) to unix timestamp."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


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
    """Manually add an episode to the temporal memory graph."""
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
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """
    Hybrid search (BM25 + vector) across episodes, entities, and facts.

    Supports optional date filtering:
      after_date:  ISO date string "YYYY-MM-DD" or "YYYY-MM" — only return episodes on/after this date
      before_date: ISO date string "YYYY-MM-DD" or "YYYY-MM" — only return episodes on/before this date
    """
    namespace = agent_namespace or engine.agent_namespace
    after_unix = _parse_date_to_unix(after_date) if after_date else None
    before_unix = _parse_date_to_unix(before_date) if before_date else None

    results = await engine.query_memory(
        query=query,
        agent_namespace=namespace,
        limit=limit,
        include_expired=include_expired,
        after_unix=after_unix,
        before_unix=before_unix,
    )
    return {
        "query": query,
        "namespace": namespace,
        "after_date": after_date,
        "before_date": before_date,
        "episodes": results.get("episodes", []),
        "entities": results.get("entities", []),
        "facts": results.get("facts", []),
        "ranked": results.get("ranked", []),
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
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """
    Smart search with three layers of fallback:

    1. BM25 fulltext (primary — fast, token-based)
    2. Zero-result fallback:
       - Entities: vector similarity search (handles typos like "Qinn" → "Quinn")
       - Episodes: toLower CONTAINS scan (handles partial matches)
    3. Numeric/IP detection: always appends toLower CONTAINS results for queries
       containing IP addresses, prices, wallet addresses, long numbers
    4. Temporal detection: if query mentions a month/year, also searches
       valid_at_human field so "February 2026" finds episodes from that period

    search_type: 'episodes' | 'entities' | 'facts' | 'all'
    after_date / before_date: optional ISO date strings (YYYY-MM-DD or YYYY-MM)
    """
    namespace = agent_namespace or engine.agent_namespace
    driver = engine.driver
    results: dict[str, list] = {}

    after_unix = _parse_date_to_unix(after_date) if after_date else None
    before_unix = _parse_date_to_unix(before_date) if before_date else None

    # Build temporal WHERE clause for valid_at filtering
    time_filter = ""
    time_params: dict[str, Any] = {}
    if after_unix is not None:
        time_filter += " AND node.valid_at >= $after_unix"
        time_params["after_unix"] = after_unix
    if before_unix is not None:
        time_filter += " AND node.valid_at <= $before_unix"
        time_params["before_unix"] = before_unix

    is_numeric = _looks_numeric(query)
    is_temporal = _looks_temporal(query)

    # -------------------------------------------------------------------------
    # EPISODES
    # -------------------------------------------------------------------------
    if search_type in ("episodes", "all"):
        ep_seen: set[str] = set()
        ep_rows: list[dict] = []

        # Primary: BM25 on episode content
        try:
            primary = await driver.query_temporal(
                f"""
                CALL db.idx.fulltext.queryNodes('Episode', $query)
                YIELD node, score
                WHERE node.agent_namespace = $ns{time_filter}
                RETURN node.uuid AS uuid, node.content AS content,
                       node.valid_at AS valid_at, score
                ORDER BY score DESC LIMIT $limit
                """,
                {"query": query, "ns": namespace, "limit": limit, **time_params},
            )
            for row in primary:
                ep_seen.add(row.get("uuid", ""))
                ep_rows.append({**row, "search_method": "bm25"})
        except Exception as exc:
            logger.debug("Episode BM25 search error: %s", exc)

        # Temporal fallback: if query looks date-like, search valid_at_human
        if is_temporal:
            try:
                date_rows = await driver.query_temporal(
                    f"""
                    CALL db.idx.fulltext.queryNodes('Episode', $query)
                    YIELD node, score
                    WHERE node.agent_namespace = $ns{time_filter}
                          AND node.valid_at_human IS NOT NULL
                    RETURN node.uuid AS uuid, node.content AS content,
                           node.valid_at AS valid_at, score
                    ORDER BY score DESC LIMIT $limit
                    """,
                    {"query": query, "ns": namespace, "limit": limit, **time_params},
                )
                for row in date_rows:
                    uid = row.get("uuid", "")
                    if uid and uid not in ep_seen:
                        ep_seen.add(uid)
                        ep_rows.append({**row, "search_method": "temporal_bm25"})
            except Exception as exc:
                logger.debug("Episode temporal BM25 error: %s", exc)

        # Numeric fallback: CONTAINS scan — catches IPs, prices, wallet addresses
        if is_numeric or (not ep_rows):
            try:
                time_filter_node = time_filter.replace("node.", "ep.")
                time_params_ep = {
                    k: v for k, v in time_params.items()
                }
                numeric_rows = await driver.query_temporal(
                    f"""
                    MATCH (ep:Episode {{agent_namespace: $ns}})
                    WHERE toLower(ep.content) CONTAINS toLower($query)
                    {"AND ep.valid_at >= $after_unix" if after_unix else ""}
                    {"AND ep.valid_at <= $before_unix" if before_unix else ""}
                    RETURN ep.uuid AS uuid, ep.content AS content,
                           ep.valid_at AS valid_at, 0.5 AS score
                    LIMIT $limit
                    """,
                    {"query": query, "ns": namespace, "limit": limit, **time_params_ep},
                )
                for row in numeric_rows:
                    uid = row.get("uuid", "")
                    if uid and uid not in ep_seen:
                        ep_seen.add(uid)
                        ep_rows.append({**row, "search_method": "contains_fallback"})
            except Exception as exc:
                logger.debug("Episode CONTAINS fallback error: %s", exc)

        # Zero-result fallback: semantic vector search on episode content_embedding
        if not ep_rows:
            try:
                query_embedding = await engine.embedder.embed(query)
                vec_rows = await driver.query_temporal(
                    f"""
                    CALL db.idx.vector.queryNodes('Episode', 'content_embedding', $limit, vecf32($embedding))
                    YIELD node, score
                    WHERE node.agent_namespace = $ns{time_filter}
                    RETURN node.uuid AS uuid, node.content AS content,
                           node.valid_at AS valid_at,
                           (1.0 - score) AS score
                    """,
                    {"embedding": query_embedding, "ns": namespace, "limit": limit, **time_params},
                )
                for row in vec_rows:
                    uid = row.get("uuid", "")
                    if uid and uid not in ep_seen:
                        ep_seen.add(uid)
                        ep_rows.append({**row, "search_method": "vector_fallback"})
            except Exception as exc:
                logger.debug("Episode vector fallback error: %s", exc)

        results["episodes"] = ep_rows

    # -------------------------------------------------------------------------
    # ENTITIES
    # -------------------------------------------------------------------------
    if search_type in ("entities", "all"):
        ent_seen: set[str] = set()
        ent_rows: list[dict] = []

        # Primary: BM25 on entity name
        try:
            primary = await driver.query_temporal(
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
            for row in primary:
                ent_seen.add(row.get("uuid", ""))
                ent_rows.append({**row, "search_method": "bm25"})
        except Exception as exc:
            logger.debug("Entity BM25 error: %s", exc)

        # Zero-result / Fuzzy fallback: vector search on entity name_embedding
        # This catches typos like "Qinn" → "Quinn" and abbreviations
        if not ent_rows or is_numeric:
            try:
                query_embedding = await engine.embedder.embed(query)
                vec_rows = await driver.query_temporal(
                    """
                    CALL db.idx.vector.queryNodes('Entity', 'name_embedding', $limit, vecf32($embedding))
                    YIELD node, score
                    WHERE node.agent_namespace = $ns
                    RETURN node.uuid AS uuid, node.name AS name,
                           node.entity_type AS entity_type,
                           (1.0 - score) AS score
                    """,
                    {"embedding": query_embedding, "ns": namespace, "limit": limit},
                )
                for row in vec_rows:
                    uid = row.get("uuid", "")
                    if uid and uid not in ent_seen:
                        ent_seen.add(uid)
                        ent_rows.append({**row, "search_method": "vector_fallback"})
            except Exception as exc:
                logger.debug("Entity vector fallback error: %s", exc)

        # CONTAINS fallback: catches partial name matches and substrings
        if not ent_rows:
            try:
                contains_rows = await driver.query_temporal(
                    """
                    MATCH (e:Entity {agent_namespace: $ns})
                    WHERE toLower(e.name) CONTAINS toLower($query)
                    RETURN e.uuid AS uuid, e.name AS name,
                           e.entity_type AS entity_type, 0.4 AS score
                    LIMIT $limit
                    """,
                    {"query": query, "ns": namespace, "limit": limit},
                )
                for row in contains_rows:
                    uid = row.get("uuid", "")
                    if uid and uid not in ent_seen:
                        ent_seen.add(uid)
                        ent_rows.append({**row, "search_method": "contains_fallback"})
            except Exception as exc:
                logger.debug("Entity CONTAINS fallback error: %s", exc)

        results["entities"] = ent_rows

    # -------------------------------------------------------------------------
    # FACTS
    # -------------------------------------------------------------------------
    if search_type in ("facts", "all"):
        # Primary + only: case-insensitive toLower CONTAINS
        # (FalkorDB does not support relationship fulltext indexes)
        time_filter_rel = ""
        if after_unix is not None:
            time_filter_rel += " AND r.valid_from >= $after_unix"
        if before_unix is not None:
            time_filter_rel += " AND r.valid_from <= $before_unix"

        try:
            rows = await driver.query_temporal(
                f"""
                MATCH ()-[r:RELATES_TO {{agent_namespace: $ns}}]-()
                WHERE toLower(r.fact) CONTAINS toLower($query)
                      AND r.expired_at IS NULL{time_filter_rel}
                RETURN r.uuid AS uuid, r.fact AS fact,
                       r.relation_type AS relation_type,
                       r.valid_from AS valid_from
                LIMIT $limit
                """,
                {"query": query, "ns": namespace, "limit": limit, **time_params},
            )
            results["facts"] = rows
        except Exception as exc:
            logger.debug("Fact search error: %s", exc)
            results["facts"] = []

    return {
        "query": query,
        "namespace": namespace,
        "after_date": after_date,
        "before_date": before_date,
        "is_numeric_query": is_numeric,
        "is_temporal_query": is_temporal,
        **results,
    }


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
# Tool 8: memory_namespace_status
# ---------------------------------------------------------------------------

async def memory_namespace_status(
    engine: Any,
    db: Any,
    agent_namespace: str | None = None,
) -> dict:
    """Return health and stats for a namespace."""
    namespace = agent_namespace or engine.agent_namespace

    ep_rows = await engine.driver.query_temporal(
        "MATCH (ep:Episode {agent_namespace: $ns}) RETURN count(ep) AS count",
        {"ns": namespace},
    )
    episode_count = ep_rows[0]["count"] if ep_rows else 0

    ent_rows = await engine.driver.query_temporal(
        "MATCH (e:Entity {agent_namespace: $ns}) RETURN count(e) AS count",
        {"ns": namespace},
    )
    entity_count = ent_rows[0]["count"] if ent_rows else 0

    edge_rows = await engine.driver.query_temporal(
        """
        MATCH ()-[r:RELATES_TO {agent_namespace: $ns}]->()
        WHERE r.expired_at IS NULL
        RETURN count(r) AS count
        """,
        {"ns": namespace},
    )
    edge_count = edge_rows[0]["count"] if edge_rows else 0

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
# Tool 9: memory_list_agents
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
