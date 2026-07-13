"""MCP tool envelope and memory_reflect unit tests (mocked engine)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openstinger.mcp.tools import memory_tools as mt

pytestmark = pytest.mark.tier1


def _engine_with_query(result: dict) -> MagicMock:
    eng = MagicMock()
    eng.agent_namespace = "default"
    eng.query_memory = AsyncMock(return_value=result)
    eng.pipeline = MagicMock()
    eng.pipeline.search = AsyncMock(return_value=result)
    eng.llm = MagicMock()
    eng.llm.complete = AsyncMock(return_value="The answer is 3. [1]")
    return eng


@pytest.mark.asyncio
async def test_memory_query_envelope_includes_confidence_and_pipeline():
    result = {
        "bm25_query": '"hello"',
        "subqueries": ["hello"],
        "conflicts": [],
        "digests": {},
        "retrieval_confidence": 0.8,
        "abstain_suggested": False,
        "tokens_used": 12,
        "items_dropped": 0,
        "pipeline": {"channels_run": ["bm25_primary"], "reranker": "none", "rrf_k": 60},
        "episodes": [{"uuid": "e1", "content": "hi"}],
        "entities": [],
        "facts": [],
        "ranked": [],
    }
    eng = _engine_with_query(result)
    out = await mt.memory_query(eng, query="hello", limit=5)
    assert out["retrieval_confidence"] == 0.8
    assert out["abstain_suggested"] is False
    assert out["pipeline"]["rrf_k"] == 60
    assert out["episodes"][0]["uuid"] == "e1"
    eng.query_memory.assert_awaited()


@pytest.mark.asyncio
async def test_memory_search_filters_by_type():
    result = {
        "bm25_query": '"x"',
        "retrieval_confidence": 0.5,
        "abstain_suggested": False,
        "pipeline": {},
        "episodes": [{"uuid": "e"}],
        "entities": [{"uuid": "n"}],
        "facts": [{"uuid": "f"}],
        "ranked": [],
        "conflicts": [],
        "tokens_used": 0,
    }
    eng = _engine_with_query(result)
    only_ep = await mt.memory_search(eng, query="x", search_type="episodes")
    assert only_ep["episodes"] and not only_ep["entities"] and not only_ep["facts"]
    only_ent = await mt.memory_search(eng, query="x", search_type="entities")
    assert only_ent["entities"] and not only_ent["episodes"]


@pytest.mark.asyncio
async def test_memory_reflect_abstains_without_llm_when_suggested():
    result = {
        "retrieval_confidence": 0.05,
        "abstain_suggested": True,
        "conflicts": [],
        "tokens_used": 0,
        "episodes": [],
    }
    eng = _engine_with_query(result)
    out = await mt.memory_reflect(eng, query="unknown thing")
    assert out["abstained"] is True
    assert "enough relevant memory" in out["answer"].lower()
    eng.llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_memory_reflect_answers_from_llm():
    result = {
        "retrieval_confidence": 0.9,
        "abstain_suggested": False,
        "conflicts": [],
        "tokens_used": 40,
        "episodes": [
            {
                "uuid": "u1",
                "source_description": "s1",
                "valid_at_human": "March 2023",
                "content": "User returned boots to Zara.",
            }
        ],
    }
    eng = _engine_with_query(result)
    out = await mt.memory_reflect(eng, query="How many items?")
    assert out["abstained"] is False
    assert "3" in out["answer"] or "answer" in out["answer"].lower()
    assert "u1" in out["supporting_uuids"]
    eng.llm.complete.assert_awaited()


@pytest.mark.asyncio
async def test_memory_reflect_maps_insufficient_sentinel():
    result = {
        "retrieval_confidence": 0.7,
        "abstain_suggested": False,
        "conflicts": [],
        "tokens_used": 10,
        "episodes": [{"uuid": "u", "content": "noise", "source_description": "s"}],
    }
    eng = _engine_with_query(result)
    eng.llm.complete = AsyncMock(return_value="INSUFFICIENT_MEMORY")
    out = await mt.memory_reflect(eng, query="q")
    assert out["abstained"] is True
