"""
Category E: SessionReader tests — byte-offset cursors, partial-line safety.

Spec: OUTDATED_DOCS_TO_BE_RENEWED/06_INTEGRATION_TEST_SCENARIOS.md §Category E
"""

from __future__ import annotations

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from tests.conftest import TEST_NAMESPACE
from openstinger.ingestion.session_reader import SessionReader


pytestmark = pytest.mark.tier1


@pytest.fixture
def temp_sessions_dir(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture
def batch_collector():
    """Collects all batches received by the callback."""
    received = []

    async def on_batch(batch):
        received.extend(batch)

    return on_batch, received


# ---------------------------------------------------------------------------
# E-1: Basic ingestion reads all complete lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e1_basic_ingestion(temp_sessions_dir, db_adapter, batch_collector):
    """SessionReader reads all complete JSONL lines."""
    on_batch, received = batch_collector

    session_file = temp_sessions_dir / "session_001.jsonl"
    lines = [
        {"content": "Hello world", "source": "conversation", "valid_at": 1700000001},
        {"content": "Second line", "source": "conversation", "valid_at": 1700000002},
    ]
    with open(session_file, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    reader = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=on_batch,
        db_adapter=db_adapter,
            session_format="simple",
        poll_interval=0.1,
        chunk_size=10,
    )

    await reader._scan_and_ingest()
    assert len(received) == 2
    assert received[0]["content"] == "Hello world"
    assert received[1]["content"] == "Second line"


# ---------------------------------------------------------------------------
# E-2: Partial line at EOF is not ingested until newline arrives
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2_partial_line_safety(temp_sessions_dir, db_adapter, batch_collector):
    """A partial last line (no trailing newline) is NOT ingested."""
    on_batch, received = batch_collector

    session_file = temp_sessions_dir / "session_002.jsonl"

    # Write one complete line + raw partial bytes (not valid JSON, no trailing \n)
    with open(session_file, "wb") as f:
        complete = json.dumps({"content": "Complete line", "source": "conversation", "valid_at": 1}) + "\n"
        # Raw partial bytes — simulates a file being written mid-line
        partial_bytes = b'{"content": "Partial lin'   # incomplete JSON, no newline
        f.write(complete.encode())
        f.write(partial_bytes)

    reader = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=on_batch,
        db_adapter=db_adapter,
            session_format="simple",
    )

    await reader._scan_and_ingest()
    assert len(received) == 1, "Only complete line should be ingested"
    assert received[0]["content"] == "Complete line"

    # Complete the partial line by appending the rest + newline
    with open(session_file, "ab") as f:
        f.write(b'e", "source": "conversation", "valid_at": 2}\n')

    await reader._scan_and_ingest()
    # Now 2 episodes total
    assert len(received) == 2
    assert received[1]["content"] == "Partial line"


# ---------------------------------------------------------------------------
# E-3: Cursor persists across reader instances (restart simulation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e3_cursor_persistence(temp_sessions_dir, db_adapter):
    """Cursor is persisted to DB; new reader instance resumes from correct position."""
    received_first = []
    received_second = []

    session_file = temp_sessions_dir / "session_003.jsonl"
    all_lines = [
        {"content": f"Episode {i}", "source": "conversation", "valid_at": 1700000000 + i}
        for i in range(6)
    ]

    with open(session_file, "w") as f:
        for line in all_lines[:3]:
            f.write(json.dumps(line) + "\n")

    # First reader: ingests lines 1-3
    async def cb1(batch):
        received_first.extend(batch)

    reader1 = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=cb1,
        db_adapter=db_adapter,
            session_format="simple",
    )
    await reader1._scan_and_ingest()
    assert len(received_first) == 3

    # Append more lines
    with open(session_file, "a") as f:
        for line in all_lines[3:]:
            f.write(json.dumps(line) + "\n")

    # Second reader (simulates restart): should only ingest lines 4-6
    async def cb2(batch):
        received_second.extend(batch)

    reader2 = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=cb2,
        db_adapter=db_adapter,
            session_format="simple",
    )
    await reader2._scan_and_ingest()
    assert len(received_second) == 3
    assert received_second[0]["content"] == "Episode 3"


# ---------------------------------------------------------------------------
# E-4: ingest_now() returns correct count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e4_ingest_now_count(temp_sessions_dir, db_adapter, batch_collector):
    """ingest_now() returns the number of episodes ingested."""
    on_batch, received = batch_collector

    session_file = temp_sessions_dir / "session_004.jsonl"
    with open(session_file, "w") as f:
        for i in range(5):
            f.write(json.dumps({"content": f"ep{i}", "source": "conversation", "valid_at": i}) + "\n")

    reader = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=on_batch,
        db_adapter=db_adapter,
            session_format="simple",
        chunk_size=2,
    )

    count = await reader.ingest_now()
    assert count == 5


# ---------------------------------------------------------------------------
# E-5: Malformed JSON lines are skipped without crashing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e5_malformed_json_skipped(temp_sessions_dir, db_adapter, batch_collector):
    """Malformed JSONL lines are logged and skipped; valid lines are still processed."""
    on_batch, received = batch_collector

    session_file = temp_sessions_dir / "session_005.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"content": "Valid 1", "source": "conversation", "valid_at": 1}) + "\n")
        f.write("this is not json\n")
        f.write(json.dumps({"content": "Valid 2", "source": "conversation", "valid_at": 2}) + "\n")

    reader = SessionReader(
        sessions_dir=temp_sessions_dir,
        agent_namespace=TEST_NAMESPACE,
        on_batch=on_batch,
        db_adapter=db_adapter,
            session_format="simple",
    )

    await reader._scan_and_ingest()
    assert len(received) == 2
    assert received[0]["content"] == "Valid 1"
    assert received[1]["content"] == "Valid 2"
