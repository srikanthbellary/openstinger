"""
FalkorDB driver — connection management, query execution, schema initialization.

Adapted from graphiti-core v0.24.0 falkordb_driver.py:
  - Removed Neo4j/Neptune/Kuzu imports
  - Explicit graph_name required (no hardcoded 'default_db')
  - FalkorDB dialect corrections (datetime as unix int, vector operator <->)
  - Added schema_init() for both temporal and knowledge graphs
  - Added async context manager support
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import falkordb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

# Temporal graph: stores Episodes, Entities, EntityEdges, EpisodicEdges
# NOTE: Index syntax verified against falkordb:latest (Redis 8.2.3):
#   - CREATE FULLTEXT INDEX FOR (n:L) ON (n.p)  ✓ supported (query via label name)
#   - CALL db.idx.fulltext.createNodeIndex(...)  ✗ creates empty index, use CREATE FULLTEXT INDEX
#   - CREATE VECTOR INDEX FOR (n:L) ON (n.p) OPTIONS {...}  ✓ supported (unnamed only)
#   - CREATE VECTOR INDEX name FOR (n:L) ON ...  ✗ named indexes NOT supported
#   - CALL db.idx.vector.createNodeIndex(...)  ✗ NOT supported (use CREATE VECTOR INDEX)
#   - CALL db.idx.fulltext.createRelationshipIndex(...)  ✗ NOT supported (skipped)
TEMPORAL_SCHEMA_QUERIES = [
    # --- Node B-tree indexes ---
    "CREATE INDEX FOR (e:Entity) ON (e.uuid)",
    "CREATE INDEX FOR (e:Entity) ON (e.name)",
    "CREATE INDEX FOR (ep:Episode) ON (ep.uuid)",
    "CREATE INDEX FOR (ep:Episode) ON (ep.agent_namespace)",
    # --- Relationship B-tree indexes ---
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.uuid)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_from)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.agent_namespace)",
    "CREATE INDEX FOR ()-[r:MENTIONS]-() ON (r.uuid)",
    # --- Full-text BM25 (nodes only — relationship fulltext not supported) ---
    "CREATE FULLTEXT INDEX FOR (e:Entity) ON (e.name)",
    "CREATE FULLTEXT INDEX FOR (ep:Episode) ON (ep.content)",
    # --- Vector indexes (no named index — FalkorDB only supports unnamed CREATE VECTOR INDEX) ---
    "CREATE VECTOR INDEX FOR (e:Entity) ON (e.name_embedding) OPTIONS {dimension: 1536, similarityFunction: 'cosine'}",
    "CREATE VECTOR INDEX FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding) OPTIONS {dimension: 1536, similarityFunction: 'cosine'}",
]

# Knowledge graph: stores vault-derived structured knowledge (Tier 2)
KNOWLEDGE_SCHEMA_QUERIES = [
    "CREATE INDEX FOR (n:Note) ON (n.uuid)",
    "CREATE INDEX FOR (n:Note) ON (n.category)",
    "CREATE INDEX FOR (n:Note) ON (n.agent_namespace)",
    "CREATE INDEX FOR (n:Note) ON (n.stale)",
    "CREATE FULLTEXT INDEX FOR (n:Note) ON (n.content)",
    "CREATE VECTOR INDEX FOR (n:Note) ON (n.content_embedding) OPTIONS {dimension: 1536, similarityFunction: 'cosine'}",
]


# ---------------------------------------------------------------------------
# Driver wrapper
# ---------------------------------------------------------------------------

class FalkorDBDriver:
    """
    Async-compatible wrapper around the synchronous falkordb client.

    FalkorDB's Python client is synchronous; we run blocking calls in the
    default thread executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        temporal_graph_name: str = "openstinger_temporal",
        knowledge_graph_name: str = "openstinger_knowledge",
    ) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.temporal_graph_name = temporal_graph_name
        self.knowledge_graph_name = knowledge_graph_name

        self._client: falkordb.FalkorDB | None = None
        self._temporal: falkordb.Graph | None = None
        self._knowledge: falkordb.Graph | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and obtain graph handles."""
        async with self._lock:
            if self._client is not None:
                return
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None, self._create_client
            )
            self._temporal = self._client.select_graph(self.temporal_graph_name)
            self._knowledge = self._client.select_graph(self.knowledge_graph_name)
            logger.info(
                "FalkorDB connected: %s:%d graphs=[%s, %s]",
                self.host, self.port,
                self.temporal_graph_name, self.knowledge_graph_name,
            )

    def _create_client(self) -> falkordb.FalkorDB:
        kwargs: dict[str, Any] = {"host": self.host, "port": self.port}
        if self.password:
            kwargs["password"] = self.password
        return falkordb.FalkorDB(**kwargs)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._client.close)
                self._client = None
                self._temporal = None
                self._knowledge = None
                logger.info("FalkorDB connection closed")

    async def ping(self) -> bool:
        """Return True if FalkorDB is reachable."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.connection.ping)  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    async def query_temporal(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the temporal graph."""
        return await self._query(self._temporal, cypher, params)

    async def query_knowledge(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the knowledge graph."""
        return await self._query(self._knowledge, cypher, params)

    async def _query(
        self,
        graph: falkordb.Graph | None,
        cypher: str,
        params: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if graph is None:
            raise RuntimeError("FalkorDB not connected — call connect() first")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: graph.query(cypher, params or {})
        )
        return self._result_to_dicts(result)

    @staticmethod
    def _result_to_dicts(result: Any) -> list[dict[str, Any]]:
        """Convert FalkorDB QueryResult to list of dicts.

        FalkorDB 1.x Python client returns headers as [[type_code, name], ...]
        not as plain strings. Row values may be Node/Relationship objects
        when the query returns full nodes (RETURN n) rather than properties.
        """
        if not hasattr(result, "result_set") or not result.result_set:
            return []
        headers = result.header
        rows = []
        for row in result.result_set:
            row_dict: dict[str, Any] = {}
            for i in range(len(headers)):
                # Extract column name — FalkorDB 1.x: [type_code, name]
                key = headers[i]
                if isinstance(key, (list, tuple)):
                    key = key[1]  # [type_code, column_name] → name string
                # Extract value — Node/Relationship → their properties dict
                val = row[i]
                if hasattr(val, "properties"):
                    val = dict(val.properties)
                row_dict[key] = val
            rows.append(row_dict)
        return rows

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """
        Create all indexes and constraints on both graphs.
        Safely ignores 'already exists' errors.
        """
        await self._init_graph_schema(self._temporal, TEMPORAL_SCHEMA_QUERIES, "temporal")
        await self._init_graph_schema(self._knowledge, KNOWLEDGE_SCHEMA_QUERIES, "knowledge")

    async def _init_graph_schema(
        self, graph: falkordb.Graph | None, queries: list[str], label: str
    ) -> None:
        if graph is None:
            raise RuntimeError("FalkorDB not connected")
        for q in queries:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda _q=q: graph.query(_q))
                logger.debug("Schema applied [%s]: %s", label, q[:60])
            except Exception as exc:
                msg = str(exc).lower()
                # FalkorDB returns errors for duplicate indexes — safe to ignore
                if "already exists" in msg or "already indexed" in msg:
                    logger.debug("Schema already exists [%s]: %s", label, q[:60])
                else:
                    logger.warning("Schema error [%s]: %s — %s", label, q[:60], exc)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "FalkorDBDriver":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

async def wait_for_falkordb(
    host: str,
    port: int,
    password: str = "",
    timeout_seconds: float = 30.0,
    retry_interval: float = 1.0,
) -> FalkorDBDriver:
    """
    Block until FalkorDB is reachable, then return a connected driver.
    Raises TimeoutError if not reachable within timeout_seconds.
    """
    deadline = time.monotonic() + timeout_seconds
    driver = FalkorDBDriver(host=host, port=port, password=password)

    while True:
        try:
            await driver.connect()
            if await driver.ping():
                logger.info("FalkorDB is ready at %s:%d", host, port)
                return driver
        except Exception as exc:
            logger.debug("FalkorDB not ready: %s", exc)

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"FalkorDB not reachable at {host}:{port} after {timeout_seconds}s"
            )
        await asyncio.sleep(retry_interval)
