"""
Operational DB unit tests — ingestion_jobs, episode_log, entity_registry, session_state.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.tier1


@pytest.mark.asyncio
async def test_create_and_retrieve_job(db_adapter):
    """Create an ingestion job, mark running, mark done."""
    job = await db_adapter.create_job(
        agent_namespace="default",
        source_file="/sessions/session.jsonl",
    )
    assert job.status == "pending"
    assert job.uuid is not None

    job.mark_running()
    await db_adapter.update_job(job)

    retrieved = await db_adapter.get_job(job.uuid)
    assert retrieved is not None
    assert retrieved.status == "running"
    assert retrieved.started_at is not None

    job.mark_done(episodes=10, entities=5, edges=8)
    await db_adapter.update_job(job)

    done = await db_adapter.get_job(job.uuid)
    assert done.status == "done"
    assert done.episodes_processed == 10


@pytest.mark.asyncio
async def test_job_mark_failed(db_adapter):
    job = await db_adapter.create_job("ns1", None)
    job.mark_failed("FalkorDB connection refused")
    await db_adapter.update_job(job)

    retrieved = await db_adapter.get_job(job.uuid)
    assert retrieved.status == "failed"
    assert "FalkorDB" in retrieved.error_message


@pytest.mark.asyncio
async def test_entity_registry_upsert_and_find(db_adapter):
    """Upsert entity, find by normalized name."""
    await db_adapter.upsert_entity(
        uuid="uuid-001",
        name="Alice Smith",
        name_normalized="alice smith",
        entity_type="PERSON",
    )

    found = await db_adapter.find_entity_by_name("alice smith")
    assert found is not None
    assert found["uuid"] == "uuid-001"
    assert found["name"] == "Alice Smith"

    # Second upsert with same normalized name should not duplicate
    await db_adapter.upsert_entity(
        uuid="uuid-002",
        name="Alice Smith (Engineer)",
        name_normalized="alice smith",
        entity_type="PERSON",
    )
    all_entities = await db_adapter.get_all_entities()
    alice_rows = [e for e in all_entities if e["name_normalized"] == "alice smith"]
    assert len(alice_rows) == 1, "Duplicate normalized names should not be inserted"


@pytest.mark.asyncio
async def test_touch_entity_increments_count(db_adapter):
    await db_adapter.upsert_entity("uuid-t1", "Bob", "bob", "PERSON")
    await db_adapter.touch_entity("uuid-t1")
    await db_adapter.touch_entity("uuid-t1")

    # Verify via get_all_entities (no direct count field exposed in dict)
    # Use internal model via session
    from openstinger.operational.models import EntityRegistryRow
    from sqlalchemy import select

    async with db_adapter._session_factory() as session:
        result = await session.execute(
            select(EntityRegistryRow).where(EntityRegistryRow.uuid == "uuid-t1")
        )
        row = result.scalar_one()
        assert row.episode_count == 2


@pytest.mark.asyncio
async def test_session_state_cursor_persistence(db_adapter):
    """Cursor set/get works correctly."""
    await db_adapter.set_cursor("agent_a", "/sessions/file1.jsonl", 1024)
    retrieved = await db_adapter.get_cursor("agent_a", "/sessions/file1.jsonl")
    assert retrieved == 1024

    # Unknown file returns 0
    zero = await db_adapter.get_cursor("agent_a", "/sessions/unknown.jsonl")
    assert zero == 0


@pytest.mark.asyncio
async def test_session_state_get_creates_if_missing(db_adapter):
    """get_session_state creates a row if namespace doesn't exist yet."""
    state = await db_adapter.get_session_state("brand_new_namespace")
    assert state is not None
    assert state.agent_namespace == "brand_new_namespace"
    assert state.session_count == 0


@pytest.mark.asyncio
async def test_episode_log(db_adapter):
    """Log an episode and retrieve it."""
    await db_adapter.log_episode(
        episode_uuid="ep-001",
        agent_namespace="default",
        source="conversation",
        entity_count=3,
        edge_count=2,
        job_uuid="job-001",
        valid_at=1700000000,
    )

    log = await db_adapter.get_episode_log("ep-001")
    assert log is not None
    assert log.entity_count == 3
    assert log.edge_count == 2
