"""
Pytest fixtures for the full OpenStinger test suite.

Fixture hierarchy:
  falkordb_config   — connection params from env
  core              — FalkorDBDriver + schema init
  temporal          — TemporalEngine (no dedup/conflict by default)
  temporal_full     — TemporalEngine with dedup + conflict resolver wired
  db_adapter        — SQLiteAdapter (in-memory :memory:)
  entity_registry   — EntityRegistry backed by db_adapter
  llm_mock          — Mock AnthropicClient (no real API calls)
  embedder_mock     — Mock OpenAIEmbedder (deterministic vectors)
  scheduler         — IngestionSchedulerRegistry
"""

from __future__ import annotations

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from openstinger.config import HarnessConfig, FalkorDBConfig
from openstinger.operational.adapter import SQLiteAdapter
from openstinger.temporal.anthropic_client import AnthropicClient
from openstinger.temporal.conflict_resolver import ConflictResolver
from openstinger.temporal.deduplicator import DeduplicationEngine
from openstinger.temporal.engine import TemporalEngine
from openstinger.temporal.entity_registry import EntityRegistry
from openstinger.temporal.falkordb_driver import FalkorDBDriver
from openstinger.temporal.nodes import EntityNode, EpisodeNode
from openstinger.temporal.openai_embedder import OpenAIEmbedder
from openstinger.ingestion.scheduler import IngestionSchedulerRegistry


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

FALKORDB_HOST = os.environ.get("TEST_FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("TEST_FALKORDB_PORT", "6379"))
FALKORDB_PASSWORD = os.environ.get("TEST_FALKORDB_PASSWORD") or os.environ.get("FALKORDB_PASSWORD", "")
TEST_NAMESPACE = "test_namespace"
TEST_GRAPH_TEMPORAL = "test_temporal"
TEST_GRAPH_KNOWLEDGE = "test_knowledge"


# ---------------------------------------------------------------------------
# Mock LLM — deterministic, no API calls
# ---------------------------------------------------------------------------

class MockAnthropicClient:
    """Deterministic mock that returns canned responses."""

    def __init__(self):
        self.model = "mock"
        self.fast_model = "mock-fast"
        # Override per test via mock.complete_json.return_value = ...
        self._complete_json_responses: list[dict] = []
        self._response_index = 0

    async def complete(self, system: str, user: str, **kwargs) -> str:
        return '{"result": "mock"}'

    async def complete_json(self, system: str, user: str, **kwargs) -> dict:
        if self._complete_json_responses:
            resp = self._complete_json_responses[self._response_index % len(self._complete_json_responses)]
            self._response_index += 1
            return resp
        # Default: no supersession, no dedup match
        if "supersedes" in system.lower() or "conflict" in system.lower():
            return {"verdict": "unrelated"}
        if "deduplication" in system.lower() or "same" in system.lower():
            return {"is_same_entity": False, "confidence": 0.1, "reasoning": "mock"}
        return {"result": "mock"}

    async def complete_with_tools(self, system: str, user: str, tools: list, **kwargs) -> dict:
        # Default: extract no entities/edges
        tool_name = tools[0]["name"] if tools else "unknown"
        if tool_name == "extract_entities":
            return {"entities": []}
        if tool_name == "extract_edges":
            return {"edges": []}
        return {}

    def set_responses(self, responses: list[dict]) -> None:
        self._complete_json_responses = responses
        self._response_index = 0


# ---------------------------------------------------------------------------
# Mock Embedder — returns deterministic vectors
# ---------------------------------------------------------------------------

class MockOpenAIEmbedder:
    """Returns deterministic unit vectors based on text hash."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        return await self._make_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self._make_vector(t) for t in texts]

    async def _make_vector(self, text: str) -> list[float]:
        # Deterministic: seed based on hash, normalize
        import hashlib
        import struct
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        dims = self.dimensions
        vec = []
        for i in range(dims):
            val = ((h >> (i % 128)) & 0xFF) / 255.0 - 0.5
            vec.append(val)
        # Normalize
        magnitude = sum(v * v for v in vec) ** 0.5
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec


# ---------------------------------------------------------------------------
# Core fixture (FalkorDB connection)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def core() -> AsyncGenerator[FalkorDBDriver, None]:
    """Session-scoped FalkorDB connection. Requires FalkorDB running.
    Skip all tests in this fixture if FalkorDB is not reachable.
    """
    import pytest
    driver = FalkorDBDriver(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        password=FALKORDB_PASSWORD,
        temporal_graph_name=TEST_GRAPH_TEMPORAL,
        knowledge_graph_name=TEST_GRAPH_KNOWLEDGE,
    )
    try:
        await driver.connect()
    except Exception as exc:
        pytest.skip(f"FalkorDB not reachable at {FALKORDB_HOST}:{FALKORDB_PORT} — {exc}")
    await driver.init_schema()
    yield driver
    await driver.close()


# ---------------------------------------------------------------------------
# Clean state between tests
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def clean_graphs(core: FalkorDBDriver) -> AsyncGenerator[None, None]:
    """Delete all nodes/edges in test graphs before each test.
    NOT autouse — only runs for tests that explicitly request it or
    that are in files marked with @pytest.mark.usefixtures("clean_graphs").
    """
    yield
    # Cleanup after test
    try:
        await core.query_temporal("MATCH (n) DETACH DELETE n")
        await core.query_knowledge("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Operational DB fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_adapter() -> AsyncGenerator[SQLiteAdapter, None]:
    """In-memory SQLite adapter."""
    adapter = SQLiteAdapter(":memory:")
    await adapter.init()
    yield adapter
    await adapter.close()


# ---------------------------------------------------------------------------
# Entity registry fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def entity_registry(db_adapter: SQLiteAdapter) -> EntityRegistry:
    reg = EntityRegistry(db_adapter)
    await reg.warmup()
    return reg


# ---------------------------------------------------------------------------
# LLM / embedder mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_mock() -> MockAnthropicClient:
    return MockAnthropicClient()


@pytest.fixture
def embedder_mock() -> MockOpenAIEmbedder:
    return MockOpenAIEmbedder()


# ---------------------------------------------------------------------------
# Temporal engine (no dedup/conflict by default)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def temporal(
    core: FalkorDBDriver,
    llm_mock: MockAnthropicClient,
    embedder_mock: MockOpenAIEmbedder,
    entity_registry: EntityRegistry,
) -> TemporalEngine:
    engine = TemporalEngine(
        driver=core,
        llm=llm_mock,
        embedder=embedder_mock,
        entity_registry=entity_registry,
        agent_namespace=TEST_NAMESPACE,
    )
    return engine


# ---------------------------------------------------------------------------
# Temporal engine with dedup + conflict resolver wired
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def temporal_full(
    core: FalkorDBDriver,
    llm_mock: MockAnthropicClient,
    embedder_mock: MockOpenAIEmbedder,
    entity_registry: EntityRegistry,
) -> TemporalEngine:
    engine = TemporalEngine(
        driver=core,
        llm=llm_mock,
        embedder=embedder_mock,
        entity_registry=entity_registry,
        agent_namespace=TEST_NAMESPACE,
    )
    dedup = DeduplicationEngine(llm=llm_mock)
    conflict = ConflictResolver(llm=llm_mock, driver=core)
    engine.set_deduplicator(dedup)
    engine.set_conflict_resolver(conflict)
    return engine


# ---------------------------------------------------------------------------
# Scheduler fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def scheduler(
    temporal: TemporalEngine,
    db_adapter: SQLiteAdapter,
) -> AsyncGenerator[IngestionSchedulerRegistry, None]:
    reg = IngestionSchedulerRegistry()
    await reg.register_agent(
        namespace=TEST_NAMESPACE,
        sessions_dir=None,  # no auto-ingestion in tests
        engine=temporal,
        db_adapter=db_adapter,
    )
    yield reg
    await reg.shutdown()


# ---------------------------------------------------------------------------
# Helper: create entity with a known fact edge
# ---------------------------------------------------------------------------

async def create_entity_edge(
    driver: FalkorDBDriver,
    source_name: str,
    target_name: str,
    relation_type: str,
    fact: str,
    namespace: str = TEST_NAMESPACE,
    valid_from: int = 1700000000,
    expired_at: int | None = None,
) -> tuple[str, str, str]:
    """
    Helper: create source entity, target entity, and edge directly in FalkorDB.
    Returns (source_uuid, target_uuid, edge_uuid).
    """
    import uuid
    src_uuid = str(uuid.uuid4())
    tgt_uuid = str(uuid.uuid4())
    edge_uuid = str(uuid.uuid4())

    await driver.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: $name, agent_namespace: $ns, entity_type: 'ENTITY', name_normalized: $norm})",
        {"uuid": src_uuid, "name": source_name, "ns": namespace, "norm": source_name.lower()},
    )
    await driver.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: $name, agent_namespace: $ns, entity_type: 'ENTITY', name_normalized: $norm})",
        {"uuid": tgt_uuid, "name": target_name, "ns": namespace, "norm": target_name.lower()},
    )

    props: dict = {
        "uuid": edge_uuid,
        "agent_namespace": namespace,
        "relation_type": relation_type,
        "fact": fact,
        "valid_from": valid_from,
        "recorded_at": valid_from,
        "episodes": [],
        "confidence": 1.0,
        "created_at": valid_from,
    }
    if expired_at is not None:
        props["expired_at"] = expired_at

    await driver.query_temporal(
        """
        MATCH (src:Entity {uuid: $src}), (tgt:Entity {uuid: $tgt})
        CREATE (src)-[r:RELATES_TO {uuid: $uuid}]->(tgt)
        SET r += $props
        """,
        {"src": src_uuid, "tgt": tgt_uuid, "uuid": edge_uuid, "props": props},
    )

    return src_uuid, tgt_uuid, edge_uuid
