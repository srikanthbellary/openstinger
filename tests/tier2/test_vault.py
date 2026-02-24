"""
Tier 2 integration tests — VaultEngine and VaultSyncEngine.

Covers:
  VE-1  Classification cycle with no episodes → no notes, cycle still logged
  VE-2  Classification cycle creates Note node in FalkorDB knowledge graph
  VE-3  Classification cycle writes to classification_log DB table
  VE-4  Classification cycle writes to vault_notes DB mirror
  VE-5  Identity note discoverable by AlignmentProfileBuilder
  VE-6  Decay marks stale notes after cutoff
  VE-7  vault_stats() returns correct category counts
  VE-8  list_notes() filters by category and excludes stale by default
  VS-1  Sync with empty vault → 0 synced, sync_log written
  VS-2  New markdown file → Note in FalkorDB + checksum in DB
  VS-3  Unchanged file on second sync → files_unchanged incremented, no re-ingest
  VS-4  Changed file on second sync → re-ingested, checksum updated
  VS-5  sync_log table written after every sync
  VS-6  ops/ directory excluded from sync
  VS-7  vault_checksums table is the canonical checksum store
"""

from __future__ import annotations

import json
import time
import uuid as uuidlib
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from tests.conftest import TEST_NAMESPACE, MockAnthropicClient, MockOpenAIEmbedder
from openstinger.gradient.alignment_profile import AlignmentProfileBuilder
from openstinger.scaffold.vault_engine import VaultEngine
from openstinger.scaffold.vault_sync import VaultSyncEngine

pytestmark = [pytest.mark.tier2, pytest.mark.integration, pytest.mark.usefixtures("clean_graphs")]


# ---------------------------------------------------------------------------
# Vault-specific LLM mock
# ---------------------------------------------------------------------------

class VaultMockLLM(MockAnthropicClient):
    """Extended mock that handles the classify_episodes tool call."""

    def __init__(self, notes_to_return: list[dict] | None = None) -> None:
        super().__init__()
        self._notes_to_return: list[dict] = notes_to_return or []
        self._evolve_responses: list[dict] = []

    def set_notes(self, notes: list[dict]) -> None:
        self._notes_to_return = notes

    def set_evolve_response(self, should_update: bool, content: str = "") -> None:
        self._evolve_responses = [{"should_update": should_update, "updated_content": content}]
        self.set_responses(self._evolve_responses)

    async def complete_with_tools(self, system: str, user: str, tools: list, **kwargs) -> dict:
        tool_name = tools[0]["name"] if tools else "unknown"
        if tool_name == "classify_episodes":
            return {"notes": self._notes_to_return}
        return await super().complete_with_tools(system, user, tools, **kwargs)


# ---------------------------------------------------------------------------
# Shared fixture: VaultEngine instance
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def vault_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vault"
    d.mkdir()
    return d


@pytest_asyncio.fixture
async def vault_llm() -> VaultMockLLM:
    return VaultMockLLM()


@pytest_asyncio.fixture
async def vault_engine(core, db_adapter, vault_llm, embedder_mock, vault_dir) -> VaultEngine:
    return VaultEngine(
        driver=core,
        llm=vault_llm,
        embedder=embedder_mock,
        db=db_adapter,
        vault_dir=vault_dir,
        agent_namespace=TEST_NAMESPACE,
    )


@pytest_asyncio.fixture
async def vault_sync(core, db_adapter, embedder_mock, vault_dir) -> VaultSyncEngine:
    return VaultSyncEngine(
        driver=core,
        embedder=embedder_mock,
        db=db_adapter,
        vault_dir=vault_dir,
        agent_namespace=TEST_NAMESPACE,
    )


# ---------------------------------------------------------------------------
# Helper: insert an episode into the temporal graph
# ---------------------------------------------------------------------------

async def insert_episode(core, content: str, namespace: str = TEST_NAMESPACE) -> str:
    ep_uuid = str(uuidlib.uuid4())
    now = int(time.time())
    await core.query_temporal(
        """
        CREATE (:Episode {
            uuid: $uuid, content: $content,
            agent_namespace: $ns, source: 'conversation',
            valid_at: $ts, created_at: $ts
        })
        """,
        {"uuid": ep_uuid, "content": content, "ns": namespace, "ts": now - 10},
    )
    return ep_uuid


# ===========================================================================
# VaultEngine Tests
# ===========================================================================

# ---------------------------------------------------------------------------
# VE-1: No episodes → cycle completes cleanly, still writes classification_log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve1_empty_cycle_logs_to_db(vault_engine, db_adapter):
    """Classification cycle with no unclassified episodes still writes to classification_log."""
    stats = await vault_engine.run_classification_cycle()

    assert stats["episodes_processed"] == 0
    assert stats["notes_created"] == 0

    # classification_log must be written even for empty cycles
    history = await db_adapter.get_classification_history(TEST_NAMESPACE, limit=5)
    assert len(history) == 1
    assert history[0].episodes_processed == 0
    assert history[0].notes_created == 0


# ---------------------------------------------------------------------------
# VE-2: Classification creates Note node in FalkorDB knowledge graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve2_cycle_creates_note_in_falkordb(vault_engine, vault_llm, core):
    """Note node appears in openstinger_knowledge after classification cycle."""
    await insert_episode(core, "I always prefer to give direct answers over lengthy explanations.")

    vault_llm.set_notes([{
        "category": "identity",
        "content": "Prefers direct answers over lengthy explanations.",
        "confidence": 0.88,
        "related_episodes": [],
    }])

    stats = await vault_engine.run_classification_cycle()
    assert stats["notes_created"] == 1

    # Verify Note node in FalkorDB knowledge graph
    rows = await core.query_knowledge(
        "MATCH (n:Note {agent_namespace: $ns}) RETURN n.category AS cat, n.content AS content, n.stale AS stale",
        {"ns": TEST_NAMESPACE},
    )
    assert len(rows) == 1
    assert rows[0]["cat"] == "identity"
    assert "direct" in rows[0]["content"]
    assert rows[0]["stale"] == 0, f"Expected stale=0, got {rows[0]['stale']}"


# ---------------------------------------------------------------------------
# VE-3: Classification cycle writes to classification_log DB table
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve3_cycle_writes_classification_log(vault_engine, vault_llm, core, db_adapter):
    """classification_log DB table captures episodes_processed, notes_created, duration_ms."""
    await insert_episode(core, "I value intellectual honesty.")
    vault_llm.set_notes([{
        "category": "identity",
        "content": "Values intellectual honesty.",
        "confidence": 0.91,
        "related_episodes": [],
    }])

    await vault_engine.run_classification_cycle()

    history = await db_adapter.get_classification_history(TEST_NAMESPACE, limit=5)
    assert len(history) == 1
    log = history[0]
    assert log.episodes_processed >= 1
    assert log.notes_created == 1
    assert log.notes_decayed == 0
    assert log.duration_ms is not None
    assert log.duration_ms >= 0


# ---------------------------------------------------------------------------
# VE-4: Classification cycle writes to vault_notes DB mirror
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve4_cycle_writes_vault_notes_db(vault_engine, vault_llm, core, db_adapter):
    """vault_notes DB table mirrors the Note nodes created in FalkorDB."""
    await insert_episode(core, "I do not make up information when I am uncertain.")
    vault_llm.set_notes([{
        "category": "constraint",
        "content": "Never fabricates information when uncertain.",
        "confidence": 0.93,
        "related_episodes": [],
    }])

    await vault_engine.run_classification_cycle()

    notes = await db_adapter.list_vault_notes(TEST_NAMESPACE, category="constraint")
    assert len(notes) == 1
    assert notes[0].category == "constraint"
    assert notes[0].stale == 0


# ---------------------------------------------------------------------------
# VE-5: Identity note discoverable by AlignmentProfileBuilder
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve5_identity_note_loads_into_alignment_profile(vault_engine, vault_llm, core):
    """After classification, AlignmentProfileBuilder can find identity notes."""
    await insert_episode(core, "I am a helpful AI assistant. I value accuracy and clarity.")
    vault_llm.set_notes([{
        "category": "identity",
        "content": "Is a helpful AI assistant that values accuracy and clarity.",
        "confidence": 0.90,
        "related_episodes": [],
    }])

    await vault_engine.run_classification_cycle()

    # AlignmentProfileBuilder reads from the knowledge graph
    builder = AlignmentProfileBuilder(driver=core, agent_namespace=TEST_NAMESPACE)
    profile = await builder.build()

    assert profile.state in ("minimal", "full"), f"Expected minimal/full, got: {profile.state}"
    assert len(profile.identity_notes) >= 1
    assert any("helpful" in n["content"].lower() or "accuracy" in n["content"].lower()
               for n in profile.identity_notes)


# ---------------------------------------------------------------------------
# VE-6: Decay marks stale notes after cutoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve6_decay_marks_stale_notes(vault_engine, core):
    """Notes with last_confirmed_at older than decay_days are marked stale."""
    # Directly create a Note in FalkorDB knowledge graph with old timestamp
    old_ts = int(time.time()) - (100 * 86400)  # 100 days ago (> default 90-day threshold)
    note_uuid = str(uuidlib.uuid4())
    await core.query_knowledge(
        """
        CREATE (n:Note {
            uuid: $uuid, agent_namespace: $ns, category: 'domain',
            content: 'Old domain knowledge.', stale: 0,
            created_at: $ts, updated_at: $ts, last_confirmed_at: $ts
        })
        """,
        {"uuid": note_uuid, "ns": TEST_NAMESPACE, "ts": old_ts},
    )

    # Run cycle — decay operation should flag this note
    # Override decay_days to 90 (default) — the note is 100 days old
    vault_engine.decay_days = 90
    stats = await vault_engine.run_classification_cycle()
    assert stats["notes_decayed"] >= 1

    # Verify note is now stale in FalkorDB
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: $uuid}) RETURN n.stale AS stale",
        {"uuid": note_uuid},
    )
    assert rows[0]["stale"] == 1, f"Expected stale=1, got {rows[0]['stale']}"


# ---------------------------------------------------------------------------
# VE-7: vault_stats returns correct category counts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve7_vault_stats(vault_engine, vault_llm, core):
    """vault_stats() returns active/stale counts per category."""
    await insert_episode(core, "I prefer concise code over verbose code.")
    await insert_episode(core, "I know Python, JavaScript, and Rust.")

    vault_llm.set_notes([
        {"category": "preference", "content": "Prefers concise code.", "confidence": 0.85, "related_episodes": []},
        {"category": "domain", "content": "Knows Python, JavaScript, Rust.", "confidence": 0.87, "related_episodes": []},
    ])

    await vault_engine.run_classification_cycle()

    stats = await vault_engine.get_vault_stats()

    assert "preference" in stats
    assert "domain" in stats
    assert stats["preference"]["active"] >= 1
    assert stats["domain"]["active"] >= 1
    # No stale yet
    assert stats["preference"]["stale"] == 0


# ---------------------------------------------------------------------------
# VE-8: list_notes filters correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ve8_list_notes_filters(vault_engine, vault_llm, core):
    """list_notes() filters by category and excludes stale notes by default."""
    await insert_episode(core, "I break problems into sub-problems.")
    await insert_episode(core, "I value being honest.")

    vault_llm.set_notes([
        {"category": "methodology", "content": "Breaks problems into sub-problems.", "confidence": 0.85, "related_episodes": []},
        {"category": "identity", "content": "Values honesty.", "confidence": 0.90, "related_episodes": []},
    ])

    await vault_engine.run_classification_cycle()

    all_notes = await vault_engine.list_notes()
    assert len(all_notes) == 2

    identity_only = await vault_engine.list_notes(category="identity")
    assert len(identity_only) == 1
    assert identity_only[0]["category"] == "identity"

    methodology_only = await vault_engine.list_notes(category="methodology")
    assert len(methodology_only) == 1


# ===========================================================================
# VaultSyncEngine Tests
# ===========================================================================

# ---------------------------------------------------------------------------
# VS-1: Sync with empty vault → 0 synced, sync_log written
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs1_empty_vault_sync(vault_sync, db_adapter):
    """Empty vault directory produces files_synced=0, still writes to sync_log."""
    stats = await vault_sync.sync()

    assert stats["files_scanned"] == 0
    assert stats["files_synced"] == 0
    assert stats["files_unchanged"] == 0

    # sync_log must be written
    async with db_adapter._session_factory() as session:
        from sqlalchemy import select
        from openstinger.operational.models import SyncLog
        result = await session.execute(
            select(SyncLog).where(SyncLog.agent_namespace == TEST_NAMESPACE)
        )
        logs = result.scalars().all()
    assert len(logs) == 1
    assert logs[0].files_synced == 0


# ---------------------------------------------------------------------------
# VS-2: New markdown file → Note in FalkorDB + checksum in DB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs2_new_file_creates_note(vault_sync, vault_dir, core, db_adapter):
    """First sync of a new vault file creates a Note in FalkorDB and stores checksum."""
    note_file = vault_dir / "test_note.md"
    note_uuid = str(uuidlib.uuid4())
    note_file.write_text(
        f"---\nuuid: {note_uuid}\ncategory: domain\n---\n\nPython is a dynamically typed language.\n",
        encoding="utf-8",
    )

    stats = await vault_sync.sync()
    assert stats["files_synced"] == 1
    assert stats["files_unchanged"] == 0

    # Note should exist in FalkorDB knowledge graph
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: $uuid}) RETURN n.content AS content, n.category AS category",
        {"uuid": note_uuid},
    )
    assert len(rows) == 1
    assert "Python" in rows[0]["content"]
    assert rows[0]["category"] == "domain"

    # Checksum should be in vault_checksums table
    checksum = await db_adapter.get_vault_checksum(TEST_NAMESPACE, str(note_file))
    assert checksum is not None
    assert len(checksum) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# VS-3: Unchanged file on second sync → files_unchanged, no re-ingest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs3_unchanged_file_not_resynced(vault_sync, vault_dir, core, db_adapter):
    """Second sync of an unchanged file increments files_unchanged, no re-ingest."""
    note_file = vault_dir / "stable_note.md"
    note_uuid = str(uuidlib.uuid4())
    note_file.write_text(
        f"---\nuuid: {note_uuid}\ncategory: preference\n---\n\nPrefers short answers.\n",
        encoding="utf-8",
    )

    # First sync — should create note
    first = await vault_sync.sync()
    assert first["files_synced"] == 1

    # Second sync — content unchanged
    second = await vault_sync.sync()
    assert second["files_synced"] == 0
    assert second["files_unchanged"] == 1

    # Still only one Note in the graph (not duplicated)
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: $uuid}) RETURN count(n) AS n",
        {"uuid": note_uuid},
    )
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# VS-4: Changed file on second sync → re-ingested, checksum updated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs4_changed_file_resynced(vault_sync, vault_dir, core, db_adapter):
    """Changed vault file is re-ingested on next sync with updated checksum."""
    note_file = vault_dir / "evolving_note.md"
    note_uuid = str(uuidlib.uuid4())
    note_file.write_text(
        f"---\nuuid: {note_uuid}\ncategory: domain\n---\n\nOriginal content.\n",
        encoding="utf-8",
    )

    first = await vault_sync.sync()
    assert first["files_synced"] == 1
    first_checksum = await db_adapter.get_vault_checksum(TEST_NAMESPACE, str(note_file))

    # Modify the file
    note_file.write_text(
        f"---\nuuid: {note_uuid}\ncategory: domain\n---\n\nUpdated content with more detail.\n",
        encoding="utf-8",
    )

    second = await vault_sync.sync()
    assert second["files_synced"] == 1
    assert second["files_unchanged"] == 0

    second_checksum = await db_adapter.get_vault_checksum(TEST_NAMESPACE, str(note_file))
    assert second_checksum != first_checksum

    # FalkorDB Note should reflect updated content
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: $uuid}) RETURN n.content AS content",
        {"uuid": note_uuid},
    )
    assert "Updated" in rows[0]["content"]


# ---------------------------------------------------------------------------
# VS-5: sync_log written after every sync
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs5_sync_log_written(vault_sync, vault_dir, db_adapter):
    """Every sync() call writes one row to sync_log table."""
    from sqlalchemy import select
    from openstinger.operational.models import SyncLog

    # Three syncs
    for i in range(3):
        f = vault_dir / f"note_{i}.md"
        f.write_text(f"---\nuuid: {uuidlib.uuid4()}\ncategory: domain\n---\n\nContent {i}.\n")
        await vault_sync.sync()

    async with db_adapter._session_factory() as session:
        result = await session.execute(
            select(SyncLog).where(SyncLog.agent_namespace == TEST_NAMESPACE)
        )
        logs = result.scalars().all()

    assert len(logs) == 3
    # Last sync scanned 3 files
    assert logs[-1].files_scanned == 3


# ---------------------------------------------------------------------------
# VS-6: ops/ directory excluded from sync
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs6_ops_dir_excluded(vault_sync, vault_dir, core):
    """Files inside vault/ops/ are never synced to FalkorDB."""
    ops_dir = vault_dir / "ops"
    ops_dir.mkdir(exist_ok=True)
    ops_note = ops_dir / "scratch.md"
    ops_note.write_text("---\nuuid: scratch-001\ncategory: domain\n---\n\nScratch content.\n")

    # Also create a normal file
    normal_note = vault_dir / "normal.md"
    note_uuid = str(uuidlib.uuid4())
    normal_note.write_text(
        f"---\nuuid: {note_uuid}\ncategory: domain\n---\n\nNormal note.\n"
    )

    stats = await vault_sync.sync()

    # Only normal_note should be synced (ops excluded)
    assert stats["files_scanned"] == 1  # ops/ not counted
    assert stats["files_synced"] == 1

    # ops content must NOT be in FalkorDB
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: 'scratch-001'}) RETURN n",
        {},
    )
    assert len(rows) == 0

    # Normal note IS in FalkorDB
    rows = await core.query_knowledge(
        "MATCH (n:Note {uuid: $uuid}) RETURN n",
        {"uuid": note_uuid},
    )
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# VS-7: vault_checksums is the canonical checksum store (not session_state JSON)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vs7_checksums_in_canonical_table(vault_sync, vault_dir, db_adapter):
    """Checksums are stored in vault_checksums table, not in session_state JSON."""
    note_file = vault_dir / "canon_note.md"
    note_uuid = str(uuidlib.uuid4())
    note_file.write_text(
        f"---\nuuid: {note_uuid}\ncategory: methodology\n---\n\nI approach things step by step.\n"
    )

    await vault_sync.sync()

    # vault_checksums table must have the entry
    checksum = await db_adapter.get_vault_checksum(TEST_NAMESPACE, str(note_file))
    assert checksum is not None

    # session_state JSON must NOT have checksum keys (they moved to vault_checksums)
    state = await db_adapter.get_session_state(TEST_NAMESPACE)
    cursors = state.get_cursors()
    checksum_keys = [k for k in cursors if k.startswith("checksum:")]
    assert len(checksum_keys) == 0, (
        "Checksums should be in vault_checksums table, not session_state JSON. "
        f"Found: {checksum_keys}"
    )
