"""
Tier 1 MCP tool handlers — 12 tools (v0.9: +memory_wake_up).

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
 10. memory_delete         — permanently delete an episode (+ prune orphaned entities)
 11. memory_update         — update episode content and re-index

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

from openstinger.temporal.search_utils import sanitize_bm25_query

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

    # v0.9: Write Policy check — detect near-duplicate before writing
    write_policy = getattr(engine, "_write_policy", None)
    if write_policy is not None:
        try:
            from openstinger.temporal.write_policy import WriteOp
            decision = await write_policy.evaluate(
                op_type         = "add",
                content         = content,
                agent_namespace = namespace,
                engine          = engine,
            )
            if decision.op == WriteOp.NOOP:
                return {
                    "status":        "noop",
                    "reason":        decision.reason,
                    "existing_uuid": decision.existing_uuid,
                    "namespace":     namespace,
                }
        except Exception as _wp_exc:
            logger.debug("write_policy check failed (fail-open): %s", _wp_exc)

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
    max_tokens: int | None = None,
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
        max_tokens=max_tokens,
    )
    return {
        "query": query,
        "bm25_query": results.get("bm25_query"),
        "subqueries": results.get("subqueries"),
        "conflicts": results.get("conflicts", []),
        "digests": results.get("digests", {}),
        "retrieval_confidence": results.get("retrieval_confidence", 0.0),
        "abstain_suggested": results.get("abstain_suggested", False),
        "tokens_used": results.get("tokens_used", 0),
        "items_dropped": results.get("items_dropped", 0),
        "pipeline": results.get("pipeline", {}),
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
    max_tokens: int | None = None,
) -> dict:
    """
    Unified search via RetrievalPipeline (same core path as memory_query).

    search_type: 'episodes' | 'entities' | 'facts' | 'all'
    """
    namespace = agent_namespace or engine.agent_namespace
    after_unix = _parse_date_to_unix(after_date) if after_date else None
    before_unix = _parse_date_to_unix(before_date) if before_date else None
    full = await engine.query_memory(
        query=query,
        agent_namespace=namespace,
        limit=limit,
        after_unix=after_unix,
        before_unix=before_unix,
        max_tokens=max_tokens,
    )
    out = {
        "query": query,
        "bm25_query": full.get("bm25_query"),
        "retrieval_confidence": full.get("retrieval_confidence", 0.0),
        "abstain_suggested": full.get("abstain_suggested", False),
        "pipeline": full.get("pipeline", {}),
        "namespace": namespace,
        "after_date": after_date,
        "before_date": before_date,
        "episodes": full.get("episodes", []) if search_type in ("episodes", "all") else [],
        "entities": full.get("entities", []) if search_type in ("entities", "all") else [],
        "facts": full.get("facts", []) if search_type in ("facts", "all") else [],
        "ranked": full.get("ranked", []),
        "conflicts": full.get("conflicts", []),
        "tokens_used": full.get("tokens_used", 0),
    }
    return out


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


# ---------------------------------------------------------------------------
# Tool 10: memory_delete
# ---------------------------------------------------------------------------

async def memory_delete(
    engine: Any,
    db: Any,
    episode_uuid: str,
) -> dict:
    """
    Permanently delete an episode from the temporal memory graph.

    Cascade-removes any Entity nodes that become orphaned after deletion
    (entities with no remaining MENTIONS relationships).

    Also removes the episode record from the operational DB.
    """
    result = await engine.delete_episode(episode_uuid)

    if result.get("deleted"):
        try:
            await db.delete_episode(episode_uuid)
        except Exception as exc:
            logger.debug("Operational DB episode delete skipped or failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Tool 11: memory_update
# ---------------------------------------------------------------------------

async def memory_update(
    engine: Any,
    episode_uuid: str,
    new_content: str,
) -> dict:
    """
    Update the content of an existing episode and re-index it.

    Re-embeds the new content for semantic search and runs entity extraction
    diff to add any new entities introduced by the updated content.
    Existing entity relationships are preserved.
    """
    return await engine.update_episode(episode_uuid, new_content)


# ---------------------------------------------------------------------------
# Tool 13: memory_reflect (v0.10 wave 2)
# ---------------------------------------------------------------------------

async def memory_reflect(
    engine: Any,
    query: str,
    max_tokens: int = 2000,
    agent_namespace: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """
    Retrieve then reason: answer from memory with optional abstention.

    Fair retrieval scores should still use memory_query; this is the product QA path.
    """
    from openstinger.temporal.prompts.reflect import REFLECT_SYSTEM, build_reflect_user

    namespace = agent_namespace or engine.agent_namespace
    after_unix = _parse_date_to_unix(after_date) if after_date else None
    before_unix = _parse_date_to_unix(before_date) if before_date else None

    retrieved = await engine.pipeline.search(
        query,
        agent_namespace=namespace,
        limit=10,
        after_unix=after_unix,
        before_unix=before_unix,
        max_tokens=max_tokens,
        candidate_multiplier=4,
    )
    if retrieved.get("abstain_suggested"):
        return {
            "answer": "I do not have enough relevant memory to answer.",
            "confidence": retrieved.get("retrieval_confidence", 0.0),
            "abstained": True,
            "supporting_uuids": [],
            "conflicts_considered": retrieved.get("conflicts", []),
            "tokens_retrieved": retrieved.get("tokens_used", 0),
            "retrieval_confidence": retrieved.get("retrieval_confidence", 0.0),
        }

    blocks = []
    uuids = []
    for i, ep in enumerate(retrieved.get("episodes") or [], 1):
        sid = ep.get("source_description") or ep.get("uuid") or ""
        when = ep.get("valid_at_human") or ""
        label = ep.get("recency_label") or ""
        header = f"[{i}] {sid}"
        if when:
            header += f" | {when}"
        if label:
            header += f" | {str(label).upper()}"
        blocks.append(f"{header}\n{ep.get('content') or ''}")
        if ep.get("uuid"):
            uuids.append(ep["uuid"])

    user = build_reflect_user(query, blocks, retrieved.get("conflicts"))
    answer = ""
    try:
        llm = engine.llm
        if hasattr(llm, "complete"):
            answer = await llm.complete(system=REFLECT_SYSTEM, user=user, use_fast_model=True)
        elif hasattr(llm, "generate"):
            answer = await llm.generate(f"{REFLECT_SYSTEM}\n\n{user}")
        else:
            answer = "INSUFFICIENT_MEMORY"
    except Exception as exc:
        logger.warning("memory_reflect LLM failed: %s", exc)
        answer = "INSUFFICIENT_MEMORY"

    text = (answer or "").strip()
    abstained = "INSUFFICIENT_MEMORY" in text.upper() or not text
    if abstained:
        text = "I do not have enough relevant memory to answer."
    return {
        "answer": text,
        "confidence": retrieved.get("retrieval_confidence", 0.0),
        "abstained": abstained,
        "supporting_uuids": uuids,
        "conflicts_considered": retrieved.get("conflicts", []),
        "tokens_retrieved": retrieved.get("tokens_used", 0),
        "retrieval_confidence": retrieved.get("retrieval_confidence", 0.0),
    }
