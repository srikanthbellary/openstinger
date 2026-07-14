"""
Temporal engine — the core of Tier 1.

Adapted from graphiti-core v0.24.0 graphiti.py (renamed engine.py):
  - Import paths: graphiti_core.* → openstinger.temporal.*
  - Explicit agent_namespace on all operations
  - FalkorDB dialect (unix timestamps, <-> vector operator, no APOC)
  - Deduplication runs before conflict resolution (not after)
  - EntityRegistry integration for cross-engine UUID coherence
  - Ontology preprocessing hook (placeholder, called before episode extraction)
"""

from __future__ import annotations

import asyncio
import logging
import uuid as uuid_lib
from datetime import datetime, timezone
from typing import Any, Optional

# Month names for BM25-indexable date string on episodes
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

from openstinger.temporal.anthropic_client import AnthropicClient
from openstinger.temporal.edges import EntityEdge, EpisodicEdge
from openstinger.temporal.entity_registry import EntityRegistry
from openstinger.temporal.falkordb_driver import FalkorDBDriver
from openstinger.temporal.nodes import EntityNode, EpisodeNode
from openstinger.temporal.openai_embedder import OpenAIEmbedder
from openstinger.temporal.prompts.extraction import (
    EXTRACT_ENTITIES_SYSTEM,
    EXTRACT_EDGES_SYSTEM,
    EXTRACT_STATEMENTS_SYSTEM,
    build_extract_entities_user,
    build_extract_edges_user,
    build_extract_statements_user,
)
from openstinger.temporal.search_utils import (
    sanitize_bm25_query,
)
from openstinger.config import RetrievalConfig
from openstinger.search.pipeline import RetrievalPipeline


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt schemas (tool_use)
# ---------------------------------------------------------------------------

EXTRACT_ENTITIES_TOOL = {
    "name": "extract_entities",
    "description": "Extract named entities from the episode content",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": ["PERSON", "ORG", "CONCEPT", "LOCATION", "EVENT", "ENTITY"],
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["name", "entity_type", "summary"],
                },
            }
        },
        "required": ["entities"],
    },
}

EXTRACT_EDGES_TOOL = {
    "name": "extract_edges",
    "description": "Extract factual relationships between entities",
    "input_schema": {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_entity_name": {"type": "string"},
                        "target_entity_name": {"type": "string"},
                        "relation_type": {"type": "string"},
                        "fact": {"type": "string"},
                        "valid_from_iso": {
                            "type": "string",
                            "description": "ISO 8601 date when this became true, or null",
                        },
                        "valid_to_iso": {
                            "type": "string",
                            "description": "ISO 8601 date when this stopped being true, or null",
                        },
                    },
                    "required": ["source_entity_name", "target_entity_name", "relation_type", "fact"],
                },
            }
        },
        "required": ["edges"],
    },
}

EXTRACT_STATEMENTS_TOOL = {
    "name": "extract_statements",
    "description": "Distill atomic self-contained memory statements from an episode",
    "input_schema": {
        "type": "object",
        "properties": {
            "statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "preference",
                                "expertise",
                                "errand",
                                "location_update",
                            ],
                        },
                        "valid_from_iso": {
                            "type": "string",
                            "description": "ISO date or null",
                        },
                    },
                    "required": ["text", "kind"],
                },
            }
        },
        "required": ["statements"],
    },
}


# ---------------------------------------------------------------------------
# TemporalEngine
# ---------------------------------------------------------------------------

class TemporalEngine:
    """
    Core temporal memory engine.

    Responsibilities:
      1. Ingest an episode (add_episode)
      2. Extract entities and relationships via LLM
      3. Deduplicate entities (delegated to DeduplicationEngine)
      4. Detect and resolve bi-temporal conflicts (delegated to ConflictResolver)
      5. Persist to FalkorDB
      6. Hybrid search (query_memory)
    """

    def __init__(
        self,
        driver: FalkorDBDriver,
        llm: AnthropicClient,
        embedder: OpenAIEmbedder,
        entity_registry: EntityRegistry,
        agent_namespace: str = "default",
        retrieval_config: RetrievalConfig | None = None,
        extract_statements: bool = True,
        typed_min_chars: int = 80,
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.entity_registry = entity_registry
        self.agent_namespace = agent_namespace
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.extract_statements = extract_statements
        self.typed_min_chars = int(typed_min_chars)
        self._pipeline: RetrievalPipeline | None = None

        # Lazy import to avoid circular
        self._deduplicator: Optional[Any] = None
        self._conflict_resolver: Optional[Any] = None

        # v0.9 — write policy (lazy import; wired after server config is available)
        self._write_policy: Optional[Any] = None

    @property
    def pipeline(self) -> RetrievalPipeline:
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(self, self.retrieval_config)
        return self._pipeline

    def set_deduplicator(self, deduplicator: Any) -> None:
        self._deduplicator = deduplicator

    def set_conflict_resolver(self, conflict_resolver: Any) -> None:
        self._conflict_resolver = conflict_resolver

    def set_write_policy(self, write_policy: Any) -> None:  # v0.9
        self._write_policy = write_policy

    # ------------------------------------------------------------------
    # Ontology preprocessing hook
    # ------------------------------------------------------------------

    async def preprocess_ontology(self, episode: EpisodeNode) -> EpisodeNode:
        """
        Optional hook called before entity extraction.
        Override or replace for domain-specific ontology preprocessing.
        Default: no-op.
        """
        return episode

    # ------------------------------------------------------------------
    # Episode ingestion
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        source: str = "conversation",
        source_description: str = "",
        valid_at: int | None = None,
        agent_namespace: str | None = None,
    ) -> EpisodeNode:
        """
        Ingest a single episode into the temporal graph.

        Pipeline:
          1. Create EpisodeNode
          2. Preprocess ontology (hook)
          3. Extract entities via LLM
          4. Deduplicate entities
          5. Embed entities
          6. Extract edges (facts) via LLM
          7. Resolve temporal conflicts on each edge
          8. Persist all nodes and edges to FalkorDB
          9. Update EntityRegistry and operational DB

        Returns the persisted EpisodeNode.
        """
        namespace = agent_namespace or self.agent_namespace
        now_unix = int(datetime.now(timezone.utc).timestamp())

        episode = EpisodeNode(
            content=content,
            source=source,
            source_description=source_description,
            agent_namespace=namespace,
            valid_at=valid_at or now_unix,
        )

        # Step 1: Ontology preprocessing
        episode = await self.preprocess_ontology(episode)

        # Step 1b: Compute human-readable date string for BM25 temporal search
        # Stores e.g. "February 16 2026 February 2026" so queries like
        # "February 2026" or "February 16" match via fulltext index.
        try:
            dt = datetime.fromtimestamp(episode.valid_at, tz=timezone.utc)
            month = _MONTH_NAMES[dt.month]
            episode.valid_at_human = (
                f"{month} {dt.day} {dt.year} "   # "February 16 2026"
                f"{month} {dt.year} "              # "February 2026"
                f"{dt.year}"                       # "2026"
            )
        except Exception:
            episode.valid_at_human = ""

        # Step 1c: Embed episode content for semantic search
        try:
            episode.content_embedding = await self.embedder.embed(content)
        except Exception as exc:
            logger.warning("Episode embedding failed (BM25 search still works): %s", exc)

        # Step 2: Persist episode node first (so edges can reference it)
        await self._persist_episode(episode)

        # Step 3: Extract entities
        extracted_entities = await self._extract_entities(episode)

        # Step 4: Deduplicate entities
        deduped_entities: list[EntityNode] = []
        for raw_entity in extracted_entities:
            if self._deduplicator:
                resolved = await self._deduplicator.resolve(raw_entity, namespace)
            else:
                resolved = raw_entity
            canonical_uuid = await self.entity_registry.get_or_register(resolved)
            resolved.uuid = canonical_uuid
            deduped_entities.append(resolved)

        # Step 5: Embed entity names
        if deduped_entities:
            names = [e.name for e in deduped_entities]
            embeddings = await self.embedder.embed_batch(names)
            for entity, emb in zip(deduped_entities, embeddings):
                entity.name_embedding = emb

        # Step 6: Persist entities (upsert)
        for entity in deduped_entities:
            await self._upsert_entity(entity)
            await self.entity_registry.touch(entity.uuid)

        # Step 7: Extract edges (facts)
        entity_map = {e.name: e for e in deduped_entities}
        raw_edges = await self._extract_edges(episode, deduped_entities)

        # Step 8: Resolve temporal conflicts + embed facts + persist edges
        for raw_edge in raw_edges:
            source_entity = entity_map.get(raw_edge.get("source_entity_name", ""))
            target_entity = entity_map.get(raw_edge.get("target_entity_name", ""))
            if not source_entity or not target_entity:
                logger.debug("Skipping edge — unknown entity: %s", raw_edge)
                continue

            edge = EntityEdge(
                source_node_uuid=source_entity.uuid,
                target_node_uuid=target_entity.uuid,
                relation_type=raw_edge.get("relation_type", "RELATES_TO"),
                fact=raw_edge.get("fact", ""),
                agent_namespace=namespace,
                valid_from=self._iso_to_unix(raw_edge.get("valid_from_iso")) or episode.valid_at,
                valid_to=self._iso_to_unix(raw_edge.get("valid_to_iso")),
                episodes=[episode.uuid],
            )

            # Embed fact
            edge.fact_embedding = await self.embedder.embed(edge.fact)

            # Conflict resolution
            if self._conflict_resolver:
                await self._conflict_resolver.resolve(edge, namespace, self.driver)
            else:
                await self._persist_edge(edge)

            # EpisodicEdge: link episode → source and target entities
            for entity_uuid in [source_entity.uuid, target_entity.uuid]:
                ep_edge = EpisodicEdge(
                    episode_uuid=episode.uuid,
                    entity_uuid=entity_uuid,
                    agent_namespace=namespace,
                )
                await self._persist_episodic_edge(ep_edge)

        episode.entity_count = len(deduped_entities)
        episode.edge_count = len(raw_edges)

        # Step 9: Distill atomic statements (v0.10 wave 2)
        if self.extract_statements and len((content or "").strip()) >= self.typed_min_chars:
            try:
                n_stmt = await self._distill_and_persist_statements(episode)
                logger.debug("Statements distilled: %d", n_stmt)
            except Exception as exc:
                logger.warning("Statement distillation failed: %s", exc)

        logger.info("Episode ingested: %s (%d entities, %d edges)",
                    episode.uuid[:8], episode.entity_count, episode.edge_count)
        return episode

    # ------------------------------------------------------------------
    # Entity/edge extraction (LLM)
    # ------------------------------------------------------------------

    async def _extract_entities(self, episode: EpisodeNode) -> list[EntityNode]:
        try:
            result = await self.llm.complete_with_tools(
                system=EXTRACT_ENTITIES_SYSTEM,
                user=build_extract_entities_user(episode.content),
                tools=[EXTRACT_ENTITIES_TOOL],
            )
            entities = []
            for item in result.get("entities", []):
                entities.append(EntityNode(
                    name=item["name"],
                    entity_type=item.get("entity_type", "ENTITY"),
                    summary=item.get("summary", ""),
                    agent_namespace=episode.agent_namespace,
                ))
            return entities
        except Exception as exc:
            logger.warning("Entity extraction failed for episode %s: %s", episode.uuid[:8], exc)
            return []

    async def _extract_edges(
        self, episode: EpisodeNode, entities: list[EntityNode]
    ) -> list[dict]:
        if len(entities) < 2:
            return []
        entity_names = [e.name for e in entities]
        try:
            ref = None
            try:
                ref = datetime.fromtimestamp(episode.valid_at, tz=timezone.utc).date().isoformat()
            except Exception:
                ref = None
            result = await self.llm.complete_with_tools(
                system=EXTRACT_EDGES_SYSTEM,
                user=build_extract_edges_user(episode.content, entity_names, reference_time=ref),
                tools=[EXTRACT_EDGES_TOOL],
            )
            return result.get("edges", [])
        except Exception as exc:
            logger.warning("Edge extraction failed for episode %s: %s", episode.uuid[:8], exc)
            return []

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _persist_episode(self, episode: EpisodeNode) -> None:
        props = episode.to_cypher_props()
        await self.driver.query_temporal(
            """
            MERGE (ep:Episode {uuid: $uuid})
            SET ep += $props
            """,
            {"uuid": episode.uuid, "props": props},
        )

    async def _upsert_entity(self, entity: EntityNode) -> None:
        props = entity.to_cypher_props()
        await self.driver.query_temporal(
            """
            MERGE (e:Entity {uuid: $uuid})
            SET e += $props
            """,
            {"uuid": entity.uuid, "props": props},
        )

    async def _persist_edge(self, edge: EntityEdge) -> None:
        props = edge.to_cypher_props()
        await self.driver.query_temporal(
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

    async def _persist_episodic_edge(self, ep_edge: EpisodicEdge) -> None:
        props = ep_edge.to_cypher_props()
        await self.driver.query_temporal(
            """
            MATCH (ep:Episode {uuid: $ep_uuid})
            MATCH (e:Entity {uuid: $entity_uuid})
            MERGE (ep)-[r:MENTIONS {uuid: $uuid}]->(e)
            SET r += $props
            """,
            {
                "ep_uuid": ep_edge.episode_uuid,
                "entity_uuid": ep_edge.entity_uuid,
                "uuid": ep_edge.uuid,
                "props": props,
            },
        )

    async def _distill_and_persist_statements(self, episode: EpisodeNode) -> int:
        """Extract atomic Statement nodes and link via DISTILLED_TO."""
        try:
            ref = datetime.fromtimestamp(episode.valid_at, tz=timezone.utc).date().isoformat()
        except Exception:
            ref = None
        try:
            result = await self.llm.complete_with_tools(
                system=EXTRACT_STATEMENTS_SYSTEM,
                user=build_extract_statements_user(episode.content, reference_time=ref),
                tools=[EXTRACT_STATEMENTS_TOOL],
            )
        except Exception as exc:
            logger.warning("Statement LLM extract failed: %s", exc)
            return 0
        raw_statements = result.get("statements") or []
        if not raw_statements:
            return 0
        allowed_kinds = {
            "fact",
            "preference",
            "expertise",
            "errand",
            "location_update",
        }
        prepared: list[tuple[str, str, int]] = []
        for s in raw_statements:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            kind = (s.get("kind") or "fact").strip().lower()
            if kind not in allowed_kinds:
                kind = "fact"
            valid_at = episode.valid_at
            iso = s.get("valid_from_iso")
            if iso:
                try:
                    valid_at = int(
                        datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
                    )
                except Exception:
                    pass
            prepared.append((text, kind, valid_at))
            if len(prepared) >= 12:
                break
        if not prepared:
            return 0
        texts = [t for t, _, _ in prepared]
        try:
            embeddings = await self.embedder.embed_batch(texts)
        except Exception as exc:
            logger.warning("Statement embed failed: %s", exc)
            embeddings = [None] * len(texts)
        n = 0
        for (text, kind, valid_at), emb in zip(prepared, embeddings):
            sid = str(uuid_lib.uuid4())
            props = {
                "uuid": sid,
                "text": text,
                "kind": kind,
                "valid_at": valid_at,
                "namespace": episode.agent_namespace,
                "source_description": episode.source_description or "",
                "episode_uuid": episode.uuid,
                "embedding": emb or [],
                "ep_uuid": episode.uuid,
                "link_uuid": str(uuid_lib.uuid4()),
            }
            try:
                await self.driver.query_temporal(
                    """
                    CREATE (s:Statement {
                        uuid: $uuid,
                        text: $text,
                        kind: $kind,
                        valid_at: $valid_at,
                        agent_namespace: $namespace,
                        source_description: $source_description,
                        episode_uuid: $episode_uuid
                    })
                    SET s.text_embedding = vecf32($embedding)
                    WITH s
                    MATCH (ep:Episode {uuid: $ep_uuid})
                    CREATE (ep)-[:DISTILLED_TO {uuid: $link_uuid}]->(s)
                    """,
                    props,
                )
                n += 1
            except Exception as exc:
                try:
                    await self.driver.query_temporal(
                        """
                        CREATE (s:Statement {
                            uuid: $uuid,
                            text: $text,
                            kind: $kind,
                            valid_at: $valid_at,
                            agent_namespace: $namespace,
                            source_description: $source_description,
                            episode_uuid: $episode_uuid
                        })
                        WITH s
                        MATCH (ep:Episode {uuid: $ep_uuid})
                        CREATE (ep)-[:DISTILLED_TO {uuid: $link_uuid}]->(s)
                        """,
                        {k: v for k, v in props.items() if k != "embedding"},
                    )
                    n += 1
                except Exception as exc2:
                    logger.debug("Statement persist failed: %s / %s", exc, exc2)
        return n

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        agent_namespace: str | None = None,
        limit: int = 10,
        include_expired: bool = False,
        after_unix: int | None = None,
        before_unix: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Hybrid search via RetrievalPipeline (RRF fusion, graph expand, optional rerank).

        Episode rows include source_description, valid_at_human, optional recency_label.
        Returns episodes, entities, facts, ranked, conflicts, digests, retrieval_confidence.
        """
        result = await self.pipeline.search(
            query,
            agent_namespace=agent_namespace,
            limit=limit,
            include_expired=include_expired,
            after_unix=after_unix,
            before_unix=before_unix,
            max_tokens=max_tokens,
        )
        _returned_uuids = [ep["uuid"] for ep in result.get("episodes", []) if ep.get("uuid")]
        if _returned_uuids:
            asyncio.create_task(self._increment_access_counts(_returned_uuids))
        return result

    # v0.9: fire-and-forget access_count increment on returned episodes
    async def _increment_access_counts(self, uuids: list[str]) -> None:
        for uuid in uuids:
            try:
                await self.driver.query_temporal(
                    "MATCH (ep:Episode {uuid: $uuid}) "
                    "SET ep.access_count = coalesce(ep.access_count, 0) + 1",
                    {"uuid": uuid},
                )
            except Exception:
                pass  # Never block a result due to increment failure

    async def get_entity(self, uuid: str) -> Optional[dict]:
        rows = await self.driver.query_temporal(
            "MATCH (e:Entity {uuid: $uuid}) RETURN e",
            {"uuid": uuid},
        )
        return rows[0] if rows else None

    async def get_episode(self, uuid: str) -> Optional[dict]:
        rows = await self.driver.query_temporal(
            "MATCH (ep:Episode {uuid: $uuid}) RETURN ep",
            {"uuid": uuid},
        )
        return rows[0] if rows else None

    async def delete_episode(self, episode_uuid: str) -> dict[str, Any]:
        """
        Permanently delete an episode and prune orphaned entities.

        Steps:
          1. Verify the episode exists.
          2. DETACH DELETE the Episode node (removes all its relationships).
          3. Remove Entity nodes that now have no MENTIONS connections at all.

        Returns a dict with deleted status and counts.
        """
        existing = await self.get_episode(episode_uuid)
        if not existing:
            return {"deleted": False, "uuid": episode_uuid, "reason": "not found"}

        await self.driver.query_temporal(
            "MATCH (ep:Episode {uuid: $uuid}) DETACH DELETE ep",
            {"uuid": episode_uuid},
        )

        orphan_rows = await self.driver.query_temporal(
            """
            MATCH (e:Entity)
            WHERE NOT EXISTS((e)<-[:MENTIONS]-())
            RETURN e.uuid AS uuid
            """,
            {},
        )
        entities_pruned = len(orphan_rows)
        if orphan_rows:
            orphan_uuids = [r["uuid"] for r in orphan_rows]
            await self.driver.query_temporal(
                "MATCH (e:Entity) WHERE e.uuid IN $uuids DETACH DELETE e",
                {"uuids": orphan_uuids},
            )

        logger.info("Episode deleted: %s  (entities pruned: %d)", episode_uuid[:8], entities_pruned)
        return {"deleted": True, "uuid": episode_uuid, "entities_pruned": entities_pruned}

    async def update_episode(self, episode_uuid: str, new_content: str) -> dict[str, Any]:
        """
        Update the content of an existing episode and re-index it.

        Steps:
          1. Verify the episode exists.
          2. Re-embed new_content → new vector.
          3. Update content + content_embedding + updated_at on the Episode node.
          4. Run entity extraction on new_content and add any new entities
             (existing entity links are preserved).

        Returns updated episode metadata.
        """
        existing = await self.get_episode(episode_uuid)
        if not existing:
            return {"updated": False, "uuid": episode_uuid, "reason": "not found"}

        now_unix = int(datetime.now(timezone.utc).timestamp())

        try:
            new_embedding = await self.embedder.embed(new_content)
        except Exception as exc:
            logger.warning("Re-embed failed for episode %s: %s", episode_uuid[:8], exc)
            new_embedding = None

        update_params: dict[str, Any] = {
            "uuid": episode_uuid,
            "content": new_content,
            "updated_at": now_unix,
        }
        if new_embedding:
            await self.driver.query_temporal(
                """
                MATCH (ep:Episode {uuid: $uuid})
                SET ep.content = $content,
                    ep.content_embedding = $embedding,
                    ep.updated_at = $updated_at
                """,
                {**update_params, "embedding": new_embedding},
            )
        else:
            await self.driver.query_temporal(
                """
                MATCH (ep:Episode {uuid: $uuid})
                SET ep.content = $content,
                    ep.updated_at = $updated_at
                """,
                update_params,
            )

        ep_props = existing.get("ep", existing)
        namespace = ep_props.get("agent_namespace", self.agent_namespace) if isinstance(ep_props, dict) else self.agent_namespace

        new_entities_added = 0
        try:
            tmp_episode = type("_Ep", (), {
                "uuid": episode_uuid,
                "content": new_content,
                "agent_namespace": namespace,
            })()
            extracted = await self._extract_entities(tmp_episode)
            for raw_entity in extracted:
                if self._deduplicator:
                    resolved = await self._deduplicator.resolve(raw_entity, namespace)
                else:
                    resolved = raw_entity
                canonical_uuid = await self.entity_registry.get_or_register(resolved)
                resolved.uuid = canonical_uuid

                existing_entity = await self.get_entity(canonical_uuid)
                if not existing_entity:
                    try:
                        emb = await self.embedder.embed(resolved.name)
                        resolved.name_embedding = emb
                    except Exception:
                        pass
                    await self._upsert_entity(resolved)
                    new_entities_added += 1
                await self.entity_registry.touch(resolved.uuid)
        except Exception as exc:
            logger.warning("Entity re-extraction failed for episode %s: %s", episode_uuid[:8], exc)

        logger.info("Episode updated: %s  (new_entities: %d)", episode_uuid[:8], new_entities_added)
        return {
            "updated": True,
            "uuid": episode_uuid,
            "content_preview": new_content[:120],
            "updated_at": now_unix,
            "new_entities_added": new_entities_added,
            "re_embedded": new_embedding is not None,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _iso_to_unix(iso: str | None) -> Optional[int]:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso.rstrip("Z"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # v0.9 — Similarity search (prerequisite for WritePolicy)
    # ------------------------------------------------------------------

    async def find_similar_episodes(
        self,
        content:         str,
        agent_namespace: str | None = None,
        threshold:       float = 0.95,
        limit:           int = 1,
    ) -> list[dict]:
        """
        Vector similarity search for near-duplicate episode detection.
        Returns list of {uuid, score} dicts where score >= threshold.

        NOTE: FalkorDB returns cosine DISTANCE (lower = more similar).
              We convert to similarity: similarity = 1.0 - distance.
              Fail-open: returns [] on any error to avoid blocking writes.
        """
        namespace = agent_namespace or self.agent_namespace
        try:
            embedding = await self.embedder.embed(content)
            rows = await self.driver.query_temporal(
                """
                CALL db.idx.vector.queryNodes('Episode', 'content_embedding', $limit, vecf32($embedding))
                YIELD node, score
                WHERE node.agent_namespace = $namespace
                RETURN node.uuid AS uuid, (1.0 - score) AS score
                """,
                {"embedding": embedding, "namespace": namespace, "limit": limit},
            )
            return [r for r in rows if r.get("score", 0) >= threshold]
        except Exception as exc:
            logger.debug("find_similar_episodes failed (fail-open): %s", exc)
            return []

    # ------------------------------------------------------------------
    # v0.9 — memory_wake_up context builder
    # ------------------------------------------------------------------

    async def get_wake_up_context(
        self,
        agent_namespace:        str | None = None,
        max_episodes:           int   = 5,
        max_notes:              int   = 3,
        min_episode_confidence: float = 0.85,
    ) -> dict:
        """
        Returns L0 context for session boot injection.
        L0 episodes = highest access_count in this namespace at or above confidence.
        L0 vault notes = best-effort (only available at Tier 2/3 — returns [] at Tier 1).
        """
        namespace = agent_namespace or self.agent_namespace

        episode_rows = await self.driver.query_temporal(
            """
            MATCH (ep:Episode {agent_namespace: $namespace})
            WHERE ep.confidence >= $min_confidence
            RETURN ep.uuid        AS uuid,
                   ep.content     AS content,
                   ep.valid_at    AS valid_at,
                   ep.access_count AS access_count,
                   ep.confidence  AS confidence
            ORDER BY ep.access_count DESC
            LIMIT $limit
            """,
            {
                "namespace":      namespace,
                "min_confidence": min_episode_confidence,
                "limit":          max_episodes,
            },
        )

        # Best-effort vault notes (always [] at Tier 1 — Tier 2/3 override in dispatch)
        l0_notes: list[dict] = []

        total = await self._count_episodes(namespace)

        from datetime import datetime, timezone as _tz
        return {
            "agent_namespace":          namespace,
            "session_injected_at":      datetime.now(_tz.utc).isoformat(),
            "l0_episodes":              episode_rows,
            "l0_vault_notes":           l0_notes,
            "total_episodes_available": total,
            "injected_episode_count":   len(episode_rows),
            "injected_note_count":      len(l0_notes),
        }

    async def _count_episodes(self, agent_namespace: str) -> int:
        rows = await self.driver.query_temporal(
            "MATCH (ep:Episode {agent_namespace: $namespace}) RETURN count(ep) AS total",
            {"namespace": agent_namespace},
        )
        return rows[0]["total"] if rows else 0
