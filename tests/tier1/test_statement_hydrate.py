"""Hydrate statement hits to parent episodes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openstinger.config import RetrievalConfig
from openstinger.search.pipeline import RetrievalPipeline

pytestmark = pytest.mark.tier1


@pytest.mark.asyncio
async def test_hydrate_statement_promotes_parent_episode():
    engine = MagicMock()
    engine.driver = MagicMock()
    engine.driver.query_temporal = AsyncMock(
        return_value=[
            {
                "stmt_uuid": "s1",
                "stmt_text": "User prefers rooftop pools.",
                "kind": "preference",
                "uuid": "ep1",
                "content": "full episode about hotels and rooftop pools",
                "valid_at": 100,
                "valid_at_human": "Jan 1 2023",
                "source_description": "lme_session:answer_pref",
            }
        ]
    )
    pipe = RetrievalPipeline(engine, RetrievalConfig())
    rows = [
        {
            "uuid": "s1",
            "text": "User prefers rooftop pools.",
            "content": "User prefers rooftop pools.",
            "kind": "preference",
            "episode_uuid": "ep1",
            "score": 0.9,
            "fusion_score": 5.0,
        }
    ]
    out = await pipe._hydrate_statement_hits(rows)
    assert len(out) == 1
    assert out[0]["uuid"] == "ep1"
    assert out[0]["result_type"] == "episode"
    assert "rooftop" in out[0]["content"]
    assert out[0]["source_description"] == "lme_session:answer_pref"
    assert out[0]["via_statement"]
