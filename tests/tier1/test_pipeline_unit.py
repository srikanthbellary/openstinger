"""RetrievalPipeline unit tests with a mocked engine/driver."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openstinger.config import RetrievalConfig
from openstinger.search.pipeline import RetrievalPipeline

pytestmark = pytest.mark.tier1


def _make_engine():
    """Minimal engine stub: embedder + driver.query_temporal."""
    eng = MagicMock()
    eng.agent_namespace = "ns"
    eng.llm = MagicMock()
    eng.embedder = MagicMock()
    eng.embedder.embed = AsyncMock(return_value=[0.1] * 8)

    async def _query(cypher: str, params: dict | None = None):
        c = cypher.lower()
        if "statement" in c:
            return []
        if "entity" in c and "name_embedding" in c:
            return [{"uuid": "ent1", "name": "Rachel", "entity_type": "PERSON", "score": 0.2}]
        if "mentions" in c:
            return [
                {
                    "uuid": "ep_graph",
                    "content": "Rachel moved to the suburbs near Tampa.",
                    "valid_at": 200,
                    "valid_at_human": "2023",
                    "source_description": "ku",
                }
            ]
        if "relates_to" in c and "fact_embedding" in c:
            return []
        if "fulltext" in c or "contains" in c or "content_embedding" in c:
            return [
                {
                    "uuid": "ep1",
                    "content": "Rachel lived in Chicago before.",
                    "valid_at": 100,
                    "valid_at_human": "2022",
                    "source_description": "old",
                    "score": 5.0,
                },
                {
                    "uuid": "ep2",
                    "content": "Unrelated shopping trip.",
                    "valid_at": 150,
                    "valid_at_human": "2022b",
                    "source_description": "noise",
                    "score": 1.0,
                },
            ]
        if "order by ep.valid_at desc" in c:
            return [
                {
                    "uuid": "ep_recent",
                    "content": "latest note",
                    "valid_at": 300,
                    "valid_at_human": "2024",
                    "source_description": "r",
                }
            ]
        return []

    eng.driver = MagicMock()
    eng.driver.query_temporal = AsyncMock(side_effect=_query)
    return eng


@pytest.mark.asyncio
async def test_pipeline_search_returns_envelope():
    eng = _make_engine()
    pipe = RetrievalPipeline(eng, RetrievalConfig(experimental_lexicons=False, reranker="none"))
    out = await pipe.search("Where did Rachel move?", limit=5)
    assert "episodes" in out
    assert "retrieval_confidence" in out
    assert "abstain_suggested" in out
    assert "pipeline" in out
    assert out["pipeline"]["experimental_lexicons"] is False
    assert "bm25_primary" in out["pipeline"]["channels_run"]
    assert isinstance(out["conflicts"], list)
    assert "digests" in out


@pytest.mark.asyncio
async def test_pipeline_empty_namespace_abstains():
    eng = _make_engine()

    async def _empty(cypher, params=None):
        return []

    eng.driver.query_temporal = AsyncMock(side_effect=_empty)
    pipe = RetrievalPipeline(eng, RetrievalConfig())
    out = await pipe.search("anything", limit=5)
    assert out["abstain_suggested"] is True
    assert out["retrieval_confidence"] == 0.0
    assert out["episodes"] == []


@pytest.mark.asyncio
async def test_pipeline_graph_channel_can_surface_mentions():
    eng = _make_engine()
    pipe = RetrievalPipeline(eng, RetrievalConfig())
    out = await pipe.search("Rachel relocation update", limit=10)
    uuids = {e.get("uuid") for e in out["episodes"]}
    assert uuids


@pytest.mark.asyncio
async def test_pipeline_respects_max_tokens():
    eng = _make_engine()
    pipe = RetrievalPipeline(eng, RetrievalConfig())
    out = await pipe.search("Rachel", limit=10, max_tokens=15)
    assert "tokens_used" in out
    assert out["tokens_used"] >= 0
