"""
Category A: Temporal conflict resolution test scenarios (A-1 through A-8).

Spec: OUTDATED_DOCS_TO_BE_RENEWED/06_INTEGRATION_TEST_SCENARIOS.md §Category A
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from tests.conftest import (
    TEST_NAMESPACE,
    MockAnthropicClient,
    create_entity_edge,
)
from openstinger.temporal.conflict_resolver import ConflictResolver
from openstinger.temporal.edges import EntityEdge


pytestmark = [pytest.mark.tier1, pytest.mark.integration, pytest.mark.usefixtures("clean_graphs")]


# ---------------------------------------------------------------------------
# A-1: Canonical Alice Smith job change (supersession)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a1_job_change_supersession(core, llm_mock: MockAnthropicClient):
    """
    Alice Smith works at Acme Corp (Jan 2025), then changes to Beta Inc (Sep 2025).
    The new WORKS_AT edge must expire the old one.
    """
    # Setup: existing edge (Alice → Acme)
    src_uuid, tgt_uuid, old_edge_uuid = await create_entity_edge(
        driver=core,
        source_name="Alice Smith",
        target_name="Acme Corp",
        relation_type="WORKS_AT",
        fact="Alice Smith works at Acme Corp as an engineer",
        valid_from=1735689600,  # Jan 1, 2025 UTC
    )

    # Create target for new employer
    import uuid as uuidlib
    beta_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'Beta Inc', agent_namespace: $ns, entity_type: 'ORG', name_normalized: 'beta inc'})",
        {"uuid": beta_uuid, "ns": TEST_NAMESPACE},
    )

    # LLM mock: new fact supersedes old
    llm_mock.set_responses([{"verdict": "supersedes"}])

    resolver = ConflictResolver(llm=llm_mock, driver=core)
    now_ts = 1756684800  # Sep 1, 2025

    new_edge = EntityEdge(
        source_node_uuid=src_uuid,
        target_node_uuid=beta_uuid,
        relation_type="WORKS_AT",
        fact="Alice Smith works at Beta Inc as a senior engineer",
        agent_namespace=TEST_NAMESPACE,
        valid_from=now_ts,
        recorded_at=now_ts,
        episodes=["ep_001"],
    )

    await resolver.resolve(new_edge, TEST_NAMESPACE)

    # Assert: old edge is now expired
    expired_rows = await core.query_temporal(
        "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.expired_at AS expired_at",
        {"uuid": old_edge_uuid},
    )
    assert expired_rows, "Old edge should still exist (never deleted)"
    assert expired_rows[0]["expired_at"] is not None, "Old edge should be expired"

    # Assert: new edge exists and is current
    new_rows = await core.query_temporal(
        """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        RETURN r.expired_at AS expired_at, r.fact AS fact
        """,
        {"uuid": new_edge.uuid},
    )
    assert new_rows, "New edge should be persisted"
    assert new_rows[0]["expired_at"] is None, "New edge should be current"
    assert "Beta Inc" in new_rows[0]["fact"]


# ---------------------------------------------------------------------------
# A-2: Full four-episode timeline (historical backfill validation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a2_four_episode_timeline(core, llm_mock: MockAnthropicClient):
    """
    Four jobs in sequence: A→B→C→D.
    Only the last edge should be current; A, B, C should be expired.
    """
    import uuid as uuidlib

    entity_uuids = [str(uuidlib.uuid4()) for _ in range(5)]
    company_names = ["StartCo", "AlphaCorp", "BetaInc", "GammaTech", "DeltaLtd"]

    for eu, name in zip(entity_uuids, company_names):
        await core.query_temporal(
            "CREATE (:Entity {uuid: $uuid, name: $name, agent_namespace: $ns, entity_type: 'ORG', name_normalized: $norm})",
            {"uuid": eu, "name": name, "ns": TEST_NAMESPACE, "norm": name.lower()},
        )

    alice_uuid = entity_uuids[0]
    times = [1704067200, 1717200000, 1735689600, 1751328000]  # 4 timestamps
    edge_uuids = []

    llm_mock.set_responses([{"verdict": "supersedes"}] * 10)

    resolver = ConflictResolver(llm=llm_mock, driver=core)

    for i, (ts, co_uuid) in enumerate(zip(times, entity_uuids[1:])):
        edge = EntityEdge(
            source_node_uuid=alice_uuid,
            target_node_uuid=co_uuid,
            relation_type="WORKS_AT",
            fact=f"Alice works at {company_names[i + 1]}",
            agent_namespace=TEST_NAMESPACE,
            valid_from=ts,
            recorded_at=ts,
        )
        edge_uuids.append(edge.uuid)
        await resolver.resolve(edge, TEST_NAMESPACE)

    # Only the last edge should be current
    current_rows = await core.query_temporal(
        """
        MATCH (e:Entity {uuid: $alice_uuid})-[r:RELATES_TO]->()
        WHERE r.relation_type = 'WORKS_AT' AND r.expired_at IS NULL
        RETURN r.uuid AS uuid
        """,
        {"alice_uuid": alice_uuid},
    )
    assert len(current_rows) == 1, "Only one WORKS_AT edge should be current"
    assert current_rows[0]["uuid"] == edge_uuids[-1]


# ---------------------------------------------------------------------------
# A-3: Consistent facts (no false conflict)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a3_consistent_facts_no_conflict(core, llm_mock: MockAnthropicClient):
    """
    Two KNOWS edges between different pairs of people.
    No conflict should be triggered (different target nodes).
    """
    src_uuid, tgt_uuid, edge1_uuid = await create_entity_edge(
        driver=core,
        source_name="Alice Smith",
        target_name="Bob Jones",
        relation_type="KNOWS",
        fact="Alice Smith knows Bob Jones",
    )

    import uuid as uuidlib
    carol_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'Carol White', agent_namespace: $ns, entity_type: 'PERSON', name_normalized: 'carol white'})",
        {"uuid": carol_uuid, "ns": TEST_NAMESPACE},
    )

    # LLM should NOT be called (different targets → no candidates)
    llm_mock.set_responses([])  # empty — call would raise

    resolver = ConflictResolver(llm=llm_mock, driver=core)

    edge2 = EntityEdge(
        source_node_uuid=src_uuid,
        target_node_uuid=carol_uuid,
        relation_type="KNOWS",
        fact="Alice Smith knows Carol White",
        agent_namespace=TEST_NAMESPACE,
        valid_from=1700000001,
    )
    await resolver.resolve(edge2, TEST_NAMESPACE)

    # Both edges should be current
    rows = await core.query_temporal(
        """
        MATCH (e:Entity {uuid: $uuid})-[r:RELATES_TO]->()
        WHERE r.relation_type = 'KNOWS' AND r.expired_at IS NULL
        RETURN count(r) AS count
        """,
        {"uuid": src_uuid},
    )
    assert rows[0]["count"] == 2


# ---------------------------------------------------------------------------
# A-4: Historical backfill (out-of-order ingestion)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a4_historical_backfill(core, llm_mock: MockAnthropicClient):
    """
    Ingest a past fact (valid_from in the past) after a current fact exists.
    The past fact should NOT expire the current one.
    """
    src_uuid, tgt_uuid, current_edge_uuid = await create_entity_edge(
        driver=core,
        source_name="Dave",
        target_name="CurrentCorp",
        relation_type="WORKS_AT",
        fact="Dave works at CurrentCorp",
        valid_from=1735689600,  # Jan 2025
    )

    import uuid as uuidlib
    old_co_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'OldCorp', agent_namespace: $ns, entity_type: 'ORG', name_normalized: 'oldcorp'})",
        {"uuid": old_co_uuid, "ns": TEST_NAMESPACE},
    )

    # LLM: consistent (past and present can coexist)
    llm_mock.set_responses([{"verdict": "consistent"}])

    resolver = ConflictResolver(llm=llm_mock, driver=core)

    past_edge = EntityEdge(
        source_node_uuid=src_uuid,
        target_node_uuid=tgt_uuid,
        relation_type="WORKS_AT",
        fact="Dave worked at OldCorp",
        agent_namespace=TEST_NAMESPACE,
        valid_from=1700000000,  # Earlier
        valid_to=1704067200,
        recorded_at=1735000000,
    )
    await resolver.resolve(past_edge, TEST_NAMESPACE)

    # Current edge should NOT be expired
    rows = await core.query_temporal(
        "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.expired_at AS expired_at",
        {"uuid": current_edge_uuid},
    )
    assert rows[0]["expired_at"] is None, "Current edge should not be expired by historical backfill"


# ---------------------------------------------------------------------------
# A-5: Same reference_time — no false expiry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a5_same_reference_time_no_false_expiry(core, llm_mock: MockAnthropicClient):
    """
    Two WORKS_AT edges with the same valid_from timestamp should not
    falsely expire each other if they are consistent.
    """
    src_uuid, tgt1_uuid, edge1_uuid = await create_entity_edge(
        driver=core,
        source_name="Eve",
        target_name="CompanyA",
        relation_type="WORKS_AT",
        fact="Eve works at CompanyA",
        valid_from=1700000000,
    )

    # Same time, different employer
    import uuid as uuidlib
    co_b_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'CompanyB', agent_namespace: $ns, entity_type: 'ORG', name_normalized: 'companyb'})",
        {"uuid": co_b_uuid, "ns": TEST_NAMESPACE},
    )

    llm_mock.set_responses([{"verdict": "consistent"}])
    resolver = ConflictResolver(llm=llm_mock, driver=core)

    edge2 = EntityEdge(
        source_node_uuid=src_uuid,
        target_node_uuid=co_b_uuid,
        relation_type="WORKS_AT",
        fact="Eve also consults at CompanyB",
        agent_namespace=TEST_NAMESPACE,
        valid_from=1700000000,  # Same timestamp
    )
    await resolver.resolve(edge2, TEST_NAMESPACE)

    # Both edges should be current
    rows = await core.query_temporal(
        """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.agent_namespace = $ns AND r.relation_type = 'WORKS_AT'
              AND r.expired_at IS NULL
        RETURN count(r) AS count
        """,
        {"ns": TEST_NAMESPACE},
    )
    assert rows[0]["count"] == 2


# ---------------------------------------------------------------------------
# A-6: Multiple independent relationships (cross-type conflict isolation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a6_cross_type_isolation(core, llm_mock: MockAnthropicClient):
    """
    WORKS_AT conflict should not affect LIVES_IN edge for same entity.
    """
    # Create Alice → Acme (WORKS_AT) and Alice → London (LIVES_IN)
    alice_uuid, acme_uuid, works_edge = await create_entity_edge(
        driver=core,
        source_name="Alice",
        target_name="Acme",
        relation_type="WORKS_AT",
        fact="Alice works at Acme",
    )

    import uuid as uuidlib
    london_uuid = str(uuidlib.uuid4())
    lives_edge_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'London', agent_namespace: $ns, entity_type: 'LOCATION', name_normalized: 'london'})",
        {"uuid": london_uuid, "ns": TEST_NAMESPACE},
    )
    await core.query_temporal(
        """
        MATCH (src:Entity {uuid: $src}), (tgt:Entity {uuid: $tgt})
        CREATE (src)-[r:RELATES_TO {uuid: $uuid, relation_type: 'LIVES_IN',
               fact: 'Alice lives in London', valid_from: 1700000000,
               recorded_at: 1700000000, agent_namespace: $ns,
               episodes: [], confidence: 1.0, created_at: 1700000000}]->(tgt)
        """,
        {"src": alice_uuid, "tgt": london_uuid, "uuid": lives_edge_uuid, "ns": TEST_NAMESPACE},
    )

    # Supersede WORKS_AT
    beta_uuid = str(uuidlib.uuid4())
    await core.query_temporal(
        "CREATE (:Entity {uuid: $uuid, name: 'Beta', agent_namespace: $ns, entity_type: 'ORG', name_normalized: 'beta'})",
        {"uuid": beta_uuid, "ns": TEST_NAMESPACE},
    )
    llm_mock.set_responses([{"verdict": "supersedes"}])
    resolver = ConflictResolver(llm=llm_mock, driver=core)

    new_works = EntityEdge(
        source_node_uuid=alice_uuid,
        target_node_uuid=beta_uuid,
        relation_type="WORKS_AT",
        fact="Alice works at Beta",
        agent_namespace=TEST_NAMESPACE,
        valid_from=1800000000,
    )
    await resolver.resolve(new_works, TEST_NAMESPACE)

    # LIVES_IN should be unaffected
    rows = await core.query_temporal(
        "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.expired_at AS expired_at",
        {"uuid": lives_edge_uuid},
    )
    assert rows[0]["expired_at"] is None, "LIVES_IN edge should not be expired by WORKS_AT conflict"


# ---------------------------------------------------------------------------
# A-7: Point-in-time query accuracy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a8_point_in_time_query(core):
    """
    Query for edges valid at a specific point in time.
    Only edges where valid_from ≤ t ≤ valid_to (or no valid_to) are returned.
    """
    src_uuid, _, edge_uuid = await create_entity_edge(
        driver=core,
        source_name="Frank",
        target_name="Company",
        relation_type="WORKS_AT",
        fact="Frank works at Company",
        valid_from=1700000000,
    )

    query_time = 1750000000

    rows = await core.query_temporal(
        """
        MATCH (e:Entity {uuid: $src_uuid})-[r:RELATES_TO]->()
        WHERE r.valid_from <= $t
          AND (r.valid_to IS NULL OR r.valid_to >= $t)
          AND r.expired_at IS NULL
        RETURN r.uuid AS uuid
        """,
        {"src_uuid": src_uuid, "t": query_time},
    )
    assert len(rows) == 1
    assert rows[0]["uuid"] == edge_uuid
