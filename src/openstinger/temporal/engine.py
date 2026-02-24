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

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from openstinger.temporal.anthropic_client import AnthropicClient
from openstinger.temporal.edges import EntityEdge, EpisodicEdge
from openstinger.temporal.entity_registry import EntityRegistry
from openstinger.temporal.falkordb_driver import FalkorDBDriver
from openstinger.temporal.nodes import EntityNode, EpisodeNode
from openstinger.temporal.openai_embedder import OpenAIEmbedder
from openstinger.temporal.prompts.extraction import (
    EXTRACT_ENTITIES_SYSTEM,
    EXTRACT_EDGES_SYSTEM,
    build_extract_entities_user,
    build_extract_edges_user,
)

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
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.entity_registry = entity_registry
        self.agent_namespace = agent_namespace

        # Lazy import to avoid circular
        self._deduplicator: Optional[Any] = None
        self._conflict_resolver: Optional[Any] = None

    def set_deduplicator(self, deduplicator: Any) -> None:
        self._deduplicator = deduplicator

    def set_conflict_resolver(self, conflict_resolver: Any) -> None:
        self._conflict_resolver = conflict_resolver

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

        logger.info("Episode ingested: %s (%d entities, %d edges)",
                    episode.uuid[:8], len(deduped_entities), len(raw_edges))
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
            result = await self.llm.complete_with_tools(
                system=EXTRACT_EDGES_SYSTEM,
                user=build_extract_edges_user(episode.content, entity_names),
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        agent_namespace: str | None = None,
        limit: int = 10,
        include_expired: bool = False,
    ) -> dict[str, Any]:
        """
        Hybrid search: BM25 keyword + vector similarity on facts and entities.
        Returns dict with keys: episodes, entities, facts.
        """
        namespace = agent_namespace or self.agent_namespace
        query_embedding = await self.embedder.embed(query)

        # BM25 search on episode content (use label name, not index name)
        episode_rows = await self.driver.query_temporal(
            """
            CALL db.idx.fulltext.queryNodes('Episode', $query)
            YIELD node, score
            WHERE node.agent_namespace = $namespace
            RETURN node.uuid AS uuid, node.content AS content,
                   node.valid_at AS valid_at, score
            ORDER BY score DESC LIMIT $limit
            """,
            {"query": query, "namespace": namespace, "limit": limit},
        )

        # Vector search on entity names (4-arg form: label, property, count, vecf32(query))
        entity_rows = await self.driver.query_temporal(
            """
            CALL db.idx.vector.queryNodes('Entity', 'name_embedding', $limit, vecf32($embedding))
            YIELD node, score
            WHERE node.agent_namespace = $namespace
            RETURN node.uuid AS uuid, node.name AS name,
                   node.entity_type AS entity_type, score
            """,
            {"embedding": query_embedding, "namespace": namespace, "limit": limit},
        )

        # Vector search on facts (4-arg form for relationships)
        expired_filter = "" if include_expired else "AND r.expired_at IS NULL"
        fact_rows = await self.driver.query_temporal(
            f"""
            CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', $limit, vecf32($embedding))
            YIELD relationship AS r, score
            WHERE r.agent_namespace = $namespace {expired_filter}
            RETURN r.uuid AS uuid, r.fact AS fact,
                   r.relation_type AS relation_type,
                   r.valid_from AS valid_from, r.expired_at AS expired_at, score
            """,
            {"embedding": query_embedding, "namespace": namespace, "limit": limit},
        )

        return {
            "episodes": episode_rows,
            "entities": entity_rows,
            "facts": fact_rows,
        }

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
