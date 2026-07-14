"""Statement distillation unit tests (mocked LLM + driver)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openstinger.temporal.engine import TemporalEngine
from openstinger.temporal.nodes import EpisodeNode

pytestmark = pytest.mark.tier1


@pytest.mark.asyncio
async def test_distill_and_persist_statements_creates_nodes():
    driver = MagicMock()
    driver.query_temporal = AsyncMock(return_value=[])
    llm = MagicMock()
    llm.complete_with_tools = AsyncMock(
        return_value={
            "statements": [
                {
                    "text": "User prefers hotels with rooftop pools.",
                    "kind": "preference",
                    "valid_from_iso": None,
                },
                {
                    "text": "User returned boots to Zara.",
                    "kind": "errand",
                    "valid_from_iso": "2023-01-01",
                },
            ]
        }
    )
    embedder = MagicMock()
    embedder.embed_batch = AsyncMock(return_value=[[0.1], [0.2]])
    registry = MagicMock()

    eng = TemporalEngine(
        driver=driver,
        llm=llm,
        embedder=embedder,
        entity_registry=registry,
        agent_namespace="ns",
        extract_statements=True,
    )
    ep = EpisodeNode(
        content="I prefer rooftop pools. I returned boots to Zara.",
        agent_namespace="ns",
        source_description="session:1",
        valid_at=1_700_000_000,
    )
    n = await eng._distill_and_persist_statements(ep)
    assert n == 2
    assert driver.query_temporal.await_count >= 2
    llm.complete_with_tools.assert_awaited()
    call_params = driver.query_temporal.await_args_list[0].args[1]
    assert call_params.get("kind") == "preference"
    assert call_params.get("source_description") == "session:1"

@pytest.mark.asyncio
async def test_distill_skips_when_llm_returns_empty():
    driver = MagicMock()
    driver.query_temporal = AsyncMock(return_value=[])
    llm = MagicMock()
    llm.complete_with_tools = AsyncMock(return_value={"statements": []})
    embedder = MagicMock()
    eng = TemporalEngine(
        driver=driver,
        llm=llm,
        embedder=embedder,
        entity_registry=MagicMock(),
        extract_statements=True,
    )
    ep = EpisodeNode(content="hello world " * 20, agent_namespace="ns", valid_at=1)
    n = await eng._distill_and_persist_statements(ep)
    assert n == 0
    driver.query_temporal.assert_not_called()
