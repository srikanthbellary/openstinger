"""
Bi-temporal conflict resolution.

Algorithm spec: OUTDATED_DOCS_TO_BE_RENEWED/03_ALGORITHM_REFERENCE.md §1

Decision tree:
  1. Find candidate edges (same source, same target, same relation_type, current)
  2. LLM semantic check: does new fact supersede any candidate?
  3. If supersedes: expire old edge, persist new edge
  4. If consistent: add episode UUID to existing edge, no new edge
  5. If unrelated: persist new edge alongside existing
"""

from __future__ import annotations

import logging
from typing import Any

from openstinger.temporal.edges import EntityEdge
from openstinger.temporal.falkordb_driver import FalkorDBDriver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

CONFLICT_CHECK_SYSTEM = """You are a temporal fact conflict analyser.
Given an existing fact and a new fact about the same relationship between the same entities,
determine the relationship between them.

Respond with a JSON object with exactly one key "verdict" and one of these values:
  "supersedes"  — the new fact replaces the old one (e.g. job changed, address updated)
  "consistent"  — both facts are true simultaneously (no conflict)
  "unrelated"   — the facts describe different aspects; both should coexist

Be conservative: only return "supersedes" if the new fact clearly invalidates the old one."""


def _build_conflict_user(existing_fact: str, new_fact: str, relation_type: str) -> str:
    return (
        f"Relation type: {relation_type}\n\n"
        f"Existing fact: {existing_fact}\n\n"
        f"New fact: {new_fact}\n\n"
        f"Does the new fact supersede the existing fact, are they consistent, or unrelated?"
    )


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """
    Resolves temporal conflicts for EntityEdge insertion.

    Called by TemporalEngine for each extracted edge before persistence.
    """

    def __init__(self, llm: Any, driver: FalkorDBDriver) -> None:
        self.llm = llm
        self.driver = driver

    async def resolve(
        self,
        new_edge: EntityEdge,
        agent_namespace: str,
        driver: FalkorDBDriver | None = None,
    ) -> None:
        """
        Check for conflicts and persist the edge appropriately.

        Modifies FalkorDB in place:
          - Supersession: expires old edge, creates new edge
          - Consistent: appends episode UUID to existing edge, no new edge created
          - Unrelated / no candidates: creates new edge
        """
        db = driver or self.driver

        # Step 1: Find candidate edges
        candidates = await self._find_candidates(
            new_edge.source_node_uuid,
            new_edge.target_node_uuid,
            new_edge.relation_type,
            agent_namespace,
            db,
        )

        if not candidates:
            # No conflict possible — persist directly
            await self._create_edge(new_edge, db)
            return

        # Step 2: LLM semantic check for each candidate
        for candidate in candidates:
            verdict = await self._llm_check(
                existing_fact=candidate.get("fact", ""),
                new_fact=new_edge.fact,
                relation_type=new_edge.relation_type,
            )
            logger.debug(
                "Conflict check [%s]: verdict=%s | old='%s...' | new='%s...'",
                new_edge.relation_type, verdict,
                candidate.get("fact", "")[:40], new_edge.fact[:40],
            )

            if verdict == "supersedes":
                # Expire old edge, create new edge, done.
                await self._expire_edge(
                    candidate["uuid"], new_edge.recorded_at, db
                )
                await self._create_edge(new_edge, db)
                return

            elif verdict == "consistent":
                # Both facts are simultaneously true — do NOT expire existing edge.
                # Spec (03_ALGORITHM_REFERENCE.md §1.3 Edge Case B):
                #   "No edge expired. New edge created. Both remain valid_to = NULL."
                # Fall through — new edge is created after all candidates checked.
                continue

            # "unrelated" — also fall through to create new edge

        # No supersession found — create new edge alongside existing
        await self._create_edge(new_edge, db)

    # ------------------------------------------------------------------
    # FalkorDB operations
    # ------------------------------------------------------------------

    async def _find_candidates(
        self,
        source_uuid: str,
        target_uuid: str,
        relation_type: str,
        agent_namespace: str,
        db: FalkorDBDriver,
    ) -> list[dict]:
        """Find ALL current edges FROM source with same relation_type.

        Spec (03_ALGORITHM_REFERENCE.md §1.3):
          Query: All current edges FROM Alice
                 WHERE relation_type = WORKS_AT
                 AND expired_at IS NULL
          → NOT filtered by target UUID — must catch Alice→Acme when adding Alice→Beta.
        """
        rows = await db.query_temporal(
            """
            MATCH (src:Entity {uuid: $src_uuid})-[r:RELATES_TO]->(tgt:Entity)
            WHERE r.relation_type = $relation_type
              AND r.agent_namespace = $namespace
              AND r.expired_at IS NULL
            RETURN r.uuid AS uuid, r.fact AS fact,
                   r.valid_from AS valid_from, r.recorded_at AS recorded_at,
                   r.episodes AS episodes
            """,
            {
                "src_uuid": source_uuid,
                "relation_type": relation_type,
                "namespace": agent_namespace,
            },
        )
        return rows

    async def _expire_edge(
        self, edge_uuid: str, expired_at: int, db: FalkorDBDriver
    ) -> None:
        await db.query_temporal(
            """
            MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
            SET r.expired_at = $expired_at
            """,
            {"uuid": edge_uuid, "expired_at": expired_at},
        )
        logger.debug("Expired edge %s at %d", edge_uuid[:8], expired_at)

    async def _create_edge(self, edge: EntityEdge, db: FalkorDBDriver) -> None:
        props = edge.to_cypher_props()
        await db.query_temporal(
            """
            MATCH (src:Entity {uuid: $src_uuid})
            MATCH (tgt:Entity {uuid: $tgt_uuid})
            CREATE (src)-[r:RELATES_TO {uuid: $uuid}]->(tgt)
            SET r += $props
            """,
            {
                "src_uuid": edge.source_node_uuid,
                "tgt_uuid": edge.target_node_uuid,
                "uuid": edge.uuid,
                "props": props,
            },
        )

    async def _append_episode(
        self, edge_uuid: str, new_episodes: list[str], db: FalkorDBDriver
    ) -> None:
        # FalkorDB: fetch current list, merge, write back
        rows = await db.query_temporal(
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.episodes AS episodes",
            {"uuid": edge_uuid},
        )
        existing: list[str] = rows[0].get("episodes", []) if rows else []
        merged = list(set(existing) | set(new_episodes))
        await db.query_temporal(
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() SET r.episodes = $episodes",
            {"uuid": edge_uuid, "episodes": merged},
        )

    # ------------------------------------------------------------------
    # LLM check
    # ------------------------------------------------------------------

    async def _llm_check(
        self, existing_fact: str, new_fact: str, relation_type: str
    ) -> str:
        """Returns one of: supersedes | consistent | unrelated"""
        try:
            result = await self.llm.complete_json(
                system=CONFLICT_CHECK_SYSTEM,
                user=_build_conflict_user(existing_fact, new_fact, relation_type),
            )
            verdict = result.get("verdict", "unrelated").lower()
            if verdict not in ("supersedes", "consistent", "unrelated"):
                logger.warning("Unexpected conflict verdict: %s — treating as unrelated", verdict)
                return "unrelated"
            return verdict
        except Exception as exc:
            logger.warning("LLM conflict check failed: %s — treating as unrelated", exc)
            return "unrelated"
