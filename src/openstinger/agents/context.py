"""
AnonymousAgentContext — read-only access to shared knowledge for task agents.

Task agents (short-lived, anonymous, tool-use agents) should not have their
own temporal graph. They get read-only access to the entity registry and
knowledge graph via this context object.

Usage:

    ctx = AnonymousAgentContext(driver=driver, entity_registry=registry)
    entity = await ctx.get_entity("alice-smith-uuid")
    notes = await ctx.query_knowledge("What are my core values?")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AnonymousAgentContext:
    """
    Read-only context for anonymous (task/tool-use) agents.

    Provides access to:
      - Entity registry (shared entities across all named agents)
      - Knowledge graph (vault notes + document chunks)

    Does NOT have access to:
      - Any named agent's episodic memory (temporal graph)
      - Write operations of any kind
    """

    def __init__(
        self,
        driver: Any,
        entity_registry: Any,
        agent_namespace: str = "anonymous",
    ) -> None:
        self._driver = driver
        self._registry = entity_registry
        self._namespace = agent_namespace

    async def get_entity(self, uuid: str) -> Optional[dict]:
        """Fetch an entity by UUID from the shared entity registry."""
        try:
            rows = await self._driver.query_temporal(
                "MATCH (e:Entity {uuid: $uuid}) RETURN e",
                {"uuid": uuid},
            )
            return rows[0] if rows else None
        except Exception as exc:
            logger.debug("AnonymousAgentContext.get_entity error: %s", exc)
            return None

    async def search_entities(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 search across all entities (shared namespace)."""
        try:
            rows = await self._driver.query_temporal(
                """
                CALL db.idx.fulltext.queryNodes('Entity', $query)
                YIELD node, score
                RETURN node.uuid AS uuid, node.name AS name,
                       node.entity_type AS entity_type,
                       node.summary AS summary, score
                ORDER BY score DESC LIMIT $limit
                """,
                {"query": query, "limit": limit},
            )
            return rows
        except Exception as exc:
            logger.debug("AnonymousAgentContext.search_entities error: %s", exc)
            return []

    async def query_knowledge(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search the knowledge graph (vault notes + document chunks) by embedding.

        Returns a list of Note and TextChunk dicts sorted by relevance.
        """
        try:
            from openstinger.temporal.openai_embedder import OpenAIEmbedder
            # The driver doesn't carry the embedder — caller must pass it
            # or use search_knowledge_bm25 for keyword-only search
            raise NotImplementedError(
                "Use search_knowledge_bm25() or pass an embedder to the context"
            )
        except NotImplementedError:
            return await self.search_knowledge_bm25(query, limit)

    async def search_knowledge_bm25(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 keyword search across vault notes in the knowledge graph."""
        try:
            rows = await self._driver.query_knowledge(
                """
                CALL db.idx.fulltext.queryNodes('Note', $query)
                YIELD node, score
                WHERE node.stale = 0
                RETURN node.uuid AS uuid, node.category AS category,
                       node.content AS content, score
                ORDER BY score DESC LIMIT $limit
                """,
                {"query": query, "limit": limit},
            )
            return rows
        except Exception as exc:
            logger.debug("AnonymousAgentContext.search_knowledge_bm25 error: %s", exc)
            return []

    async def get_note(self, uuid: str) -> Optional[dict]:
        """Fetch a single vault note by UUID."""
        try:
            rows = await self._driver.query_knowledge(
                "MATCH (n:Note {uuid: $uuid}) RETURN n",
                {"uuid": uuid},
            )
            return rows[0] if rows else None
        except Exception as exc:
            logger.debug("AnonymousAgentContext.get_note error: %s", exc)
            return None
