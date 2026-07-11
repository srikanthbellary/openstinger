"""
Issue #2 — episode_log must record real entity_count / edge_count.

Regression: scheduler previously hard-coded both counts to 0.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from openstinger.temporal.nodes import EpisodeNode

pytestmark = pytest.mark.tier1


@pytest.mark.asyncio
async def test_process_batch_logs_real_entity_and_edge_counts(scheduler, db_adapter):
    """Scheduler must pass episode.entity_count / edge_count into log_episode."""
    namespace = "test_namespace"
    engine = scheduler._engines[namespace]

    returned = EpisodeNode(
        content="Alice met Bob at Acme.",
        source="conversation",
        agent_namespace=namespace,
        entity_count=5,
        edge_count=3,
    )
    engine.add_episode = AsyncMock(return_value=returned)

    await scheduler._process_batch(
        namespace,
        [
            {
                "content": "Alice met Bob at Acme.",
                "source": "conversation",
                "valid_at": 1740000000,
            }
        ],
    )

    log = await db_adapter.get_episode_log(returned.uuid)
    assert log is not None
    assert log.entity_count == 5
    assert log.edge_count == 3


@pytest.mark.asyncio
async def test_add_episode_sets_counts_from_extraction(
    temporal, llm_mock, clean_graphs
):
    """TemporalEngine must stamp entity_count / edge_count on the returned node."""
    async def _tools(system, user, tools, **kwargs):
        name = tools[0]["name"] if tools else ""
        if name == "extract_entities":
            return {
                "entities": [
                    {"name": "Alice", "entity_type": "PERSON"},
                    {"name": "Bob", "entity_type": "PERSON"},
                    {"name": "Acme", "entity_type": "ORGANIZATION"},
                ]
            }
        if name == "extract_edges":
            return {
                "edges": [
                    {
                        "source_entity_name": "Alice",
                        "target_entity_name": "Bob",
                        "relation_type": "MET",
                        "fact": "Alice met Bob",
                    },
                    {
                        "source_entity_name": "Alice",
                        "target_entity_name": "Acme",
                        "relation_type": "WORKS_AT",
                        "fact": "Alice works at Acme",
                    },
                ]
            }
        return {}

    llm_mock.complete_with_tools = _tools

    episode = await temporal.add_episode(
        content="Alice met Bob at Acme.",
        source="conversation",
        valid_at=1740000000,
    )

    assert episode.entity_count == 3
    assert episode.edge_count == 2
