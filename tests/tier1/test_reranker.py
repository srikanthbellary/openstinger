"""Unit tests for pluggable rerankers (v0.10 wave 2)."""

from unittest.mock import AsyncMock

import pytest

from openstinger.search.reranker import (
    NoopReranker,
    LLMReranker,
    build_reranker,
)

pytestmark = pytest.mark.tier1


@pytest.mark.asyncio
async def test_noop_preserves_order():
    items = [{"uuid": "a"}, {"uuid": "b"}]
    out = await NoopReranker().rerank("q", items)
    assert [x["uuid"] for x in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_llm_reranker_reorders_by_indices():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="1, 0")
    items = [
        {"uuid": "first", "content": "aaa"},
        {"uuid": "second", "content": "bbb"},
    ]
    out = await LLMReranker(llm).rerank("query", items)
    assert [x["uuid"] for x in out] == ["second", "first"]
    assert "rerank_score" in out[0]


@pytest.mark.asyncio
async def test_llm_reranker_falls_back_on_error():
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("boom"))
    items = [{"uuid": "a", "content": "x"}, {"uuid": "b", "content": "y"}]
    out = await LLMReranker(llm).rerank("q", items)
    assert [x["uuid"] for x in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_llm_reranker_single_item_noop():
    llm = AsyncMock()
    items = [{"uuid": "only", "content": "x"}]
    out = await LLMReranker(llm).rerank("q", items)
    assert out == items
    llm.complete.assert_not_called()


def test_build_reranker_none_and_llm():
    assert isinstance(build_reranker("none"), NoopReranker)
    assert isinstance(build_reranker("llm", llm=object()), LLMReranker)
    assert isinstance(build_reranker("llm", llm=None), NoopReranker)


def test_build_reranker_cross_encoder_without_extra_is_safe():
    # Without sentence-transformers installed, should not raise at build time
    # (CrossEncoderReranker may be returned; load happens on first call)
    r = build_reranker("cross_encoder")
    assert r is not None
