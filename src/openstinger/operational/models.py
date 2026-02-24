"""
SQLAlchemy ORM models for the operational database.

Spec: docs/05_OPERATIONAL_DB_SCHEMA.md

All 11 tables across 3 tiers (additive — each tier adds tables, never removes):

  Tier 1 (4 tables):
    ingestion_jobs   — ingestion pipeline job lifecycle tracking
    episode_log      — lightweight metadata per ingested episode
    entity_registry  — canonical name → FalkorDB UUID mapping
    session_state    — per-agent context, byte-offset cursors

  Tier 2 (4 tables):
    vault_notes       — vault note metadata mirror (not content)
    classification_log — VectraVault classification cycle log
    vault_checksums   — SHA-256 checksums for vault file change detection
    sync_log          — vault sync cycle log

  Tier 3 (3 tables):
    alignment_events  — per-output evaluation log (verdict, scores, evidence)
    drift_log         — rolling window state snapshots
    correction_log    — before/after diffs for corrected outputs

SQLite dialect:
  - UUID as TEXT
  - timestamps as INTEGER (unix seconds)
  - JSON as TEXT

PostgreSQL dialect (same models, different column types handled by SQLAlchemy):
  - UUID as UUID
  - timestamps as BIGINT
  - JSON as JSONB
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from sqlalchemy import BigInteger, Boolean, Column, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> int:
    return int(time.time())


def _uuid4() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ingestion_jobs
# ---------------------------------------------------------------------------

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), default=_uuid4, unique=True, nullable=False)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    source_file: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), default="session_jsonl")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # pending → running → done | failed

    byte_offset_start: Mapped[int] = mapped_column(BigInteger, default=0)
    byte_offset_end: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    episodes_processed: Mapped[int] = mapped_column(Integer, default=0)
    entities_created: Mapped[int] = mapped_column(Integer, default=0)
    edges_created: Mapped[int] = mapped_column(Integer, default=0)
    edges_expired: Mapped[int] = mapped_column(Integer, default=0)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[int] = mapped_column(BigInteger, default=_now)
    started_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    completed_at: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    def __init__(self, **kwargs: object) -> None:
        # SQLAlchemy column defaults apply on INSERT, not Python instantiation.
        # Set Python-level defaults explicitly so model is usable before flush.
        kwargs.setdefault("uuid", _uuid4())
        kwargs.setdefault("agent_namespace", "default")
        kwargs.setdefault("source_type", "session_jsonl")
        kwargs.setdefault("status", "pending")
        kwargs.setdefault("byte_offset_start", 0)
        kwargs.setdefault("episodes_processed", 0)
        kwargs.setdefault("entities_created", 0)
        kwargs.setdefault("edges_created", 0)
        kwargs.setdefault("edges_expired", 0)
        kwargs.setdefault("created_at", _now())
        super().__init__(**kwargs)

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = _now()

    def mark_done(
        self,
        episodes: int = 0,
        entities: int = 0,
        edges: int = 0,
        expired: int = 0,
    ) -> None:
        self.status = "done"
        self.completed_at = _now()
        self.episodes_processed = episodes
        self.entities_created = entities
        self.edges_created = edges
        self.edges_expired = expired

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.completed_at = _now()
        self.error_message = error[:2000]


# ---------------------------------------------------------------------------
# episode_log
# ---------------------------------------------------------------------------

class EpisodeLog(Base):
    __tablename__ = "episode_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    source: Mapped[str] = mapped_column(String(32), default="conversation")
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    ingestion_job_uuid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[int] = mapped_column(BigInteger, default=_now)
    valid_at: Mapped[int] = mapped_column(BigInteger, default=_now)


# ---------------------------------------------------------------------------
# entity_registry
# ---------------------------------------------------------------------------

class EntityRegistryRow(Base):
    __tablename__ = "entity_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_normalized: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), default="ENTITY")
    # JSON array of alternative name strings
    name_variants_json: Mapped[str] = mapped_column(Text, default="[]")
    episode_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[int] = mapped_column(BigInteger, default=_now)
    last_seen_at: Mapped[int] = mapped_column(BigInteger, default=_now)

    def get_name_variants(self) -> list[str]:
        return json.loads(self.name_variants_json or "[]")

    def add_name_variant(self, variant: str) -> None:
        variants = self.get_name_variants()
        if variant not in variants:
            variants.append(variant)
            self.name_variants_json = json.dumps(variants)

    def touch(self) -> None:
        self.episode_count += 1
        self.last_seen_at = _now()


# ---------------------------------------------------------------------------
# session_state
# ---------------------------------------------------------------------------

class SessionState(Base):
    __tablename__ = "session_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_namespace: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    session_count: Mapped[int] = mapped_column(Integer, default=0)
    # LLM-generated summary of recent sessions
    context_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Byte offset cursor for SessionReader
    session_file_cursor_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[int] = mapped_column(BigInteger, default=_now)
    updated_at: Mapped[int] = mapped_column(BigInteger, default=_now)

    def start_session(self) -> None:
        self.session_count += 1
        self.updated_at = _now()

    def update_summary(self, summary: str) -> None:
        self.context_summary = summary
        self.updated_at = _now()

    def get_cursors(self) -> dict:
        return json.loads(self.session_file_cursor_json or "{}")

    def set_cursor(self, file_path: str, byte_offset: int) -> None:
        cursors = self.get_cursors()
        cursors[file_path] = byte_offset
        self.session_file_cursor_json = json.dumps(cursors)
        self.updated_at = _now()


# ---------------------------------------------------------------------------
# TIER 2 — vault_notes
# ---------------------------------------------------------------------------

class VaultNote(Base):
    """Metadata mirror of vault Note nodes from FalkorDB knowledge graph."""
    __tablename__ = "vault_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # identity | domain | methodology | preference | constraint
    stale: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 0 = active, 1 = stale
    confidence: Mapped[float] = mapped_column(default=0.85)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_confirmed_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __init__(self, **kwargs: object) -> None:
        now = _now()
        kwargs.setdefault("confidence", 0.85)
        kwargs.setdefault("stale", 0)
        kwargs.setdefault("created_at", now)
        kwargs.setdefault("updated_at", now)
        kwargs.setdefault("last_confirmed_at", now)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# TIER 2 — classification_log
# ---------------------------------------------------------------------------

class ClassificationLog(Base):
    """One row per VectraVault classification cycle completion."""
    __tablename__ = "classification_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    episodes_processed: Mapped[int] = mapped_column(Integer, default=0)
    notes_created: Mapped[int] = mapped_column(Integer, default=0)
    notes_evolved: Mapped[int] = mapped_column(Integer, default=0)
    notes_decayed: Mapped[int] = mapped_column(Integer, default=0)
    mocs_updated: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completed_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("episodes_processed", 0)
        kwargs.setdefault("notes_created", 0)
        kwargs.setdefault("notes_evolved", 0)
        kwargs.setdefault("notes_decayed", 0)
        kwargs.setdefault("mocs_updated", 0)
        kwargs.setdefault("completed_at", _now())
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# TIER 2 — vault_checksums
# ---------------------------------------------------------------------------

class VaultChecksum(Base):
    """SHA-256 checksums for vault markdown files (canonical change detection store)."""
    __tablename__ = "vault_checksums"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    last_synced_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("agent_namespace", "file_path", name="uq_vault_checksums_ns_path"),
    )

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("last_synced_at", _now())
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# TIER 2 — sync_log
# ---------------------------------------------------------------------------

class SyncLog(Base):
    """One row per VaultSyncEngine sync cycle completion."""
    __tablename__ = "sync_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    files_scanned: Mapped[int] = mapped_column(Integer, default=0)
    files_synced: Mapped[int] = mapped_column(Integer, default=0)
    files_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completed_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("files_scanned", 0)
        kwargs.setdefault("files_synced", 0)
        kwargs.setdefault("files_unchanged", 0)
        kwargs.setdefault("completed_at", _now())
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# TIER 3 — alignment_events
# ---------------------------------------------------------------------------

class AlignmentEvent(Base):
    """Per-output evaluation log — full evidence stored for calibration."""
    __tablename__ = "alignment_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # pass | soft_flag | hard_block | timeout_passthrough | degraded_passthrough
    value_coherence_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    identity_consistent: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 1 = True, 0 = False, NULL = skipped
    constraint_compliant: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_safe: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    issues_json: Mapped[str] = mapped_column(Text, default="[]")
    scores_json: Mapped[str] = mapped_column(Text, default="{}")
    corrected: Mapped[int] = mapped_column(Integer, default=0)
    profile_state: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evaluated_at: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("uuid", _uuid4())
        kwargs.setdefault("issues_json", "[]")
        kwargs.setdefault("scores_json", "{}")
        kwargs.setdefault("corrected", 0)
        kwargs.setdefault("evaluated_at", _now())
        super().__init__(**kwargs)

    def get_issues(self) -> list[str]:
        return json.loads(self.issues_json or "[]")

    def get_scores(self) -> dict:
        return json.loads(self.scores_json or "{}")


# ---------------------------------------------------------------------------
# TIER 3 — drift_log
# ---------------------------------------------------------------------------

class DriftLog(Base):
    """Rolling window state snapshots — one row per alert trigger or significant change."""
    __tablename__ = "drift_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    window_size: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    mean_score: Mapped[float] = mapped_column(nullable=False)
    consecutive_flags: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_flagged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    soft_flag_rate: Mapped[float] = mapped_column(nullable=False, default=0.0)
    alert_triggered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_json: Mapped[str] = mapped_column(Text, default="[]")
    recorded_at: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("window_size", 20)
        kwargs.setdefault("consecutive_flags", 0)
        kwargs.setdefault("total_evaluated", 0)
        kwargs.setdefault("total_flagged", 0)
        kwargs.setdefault("soft_flag_rate", 0.0)
        kwargs.setdefault("alert_triggered", 0)
        kwargs.setdefault("window_json", "[]")
        kwargs.setdefault("recorded_at", _now())
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# TIER 3 — correction_log
# ---------------------------------------------------------------------------

class CorrectionLog(Base):
    """Before/after diffs for every CorrectionEngine rewrite."""
    __tablename__ = "correction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    agent_namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    alignment_event_uuid: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    original_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    corrected_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    re_eval_verdict: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    issues_json: Mapped[str] = mapped_column(Text, default="[]")
    correction_succeeded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    corrected_at: Mapped[int] = mapped_column(BigInteger, nullable=False)

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("uuid", _uuid4())
        kwargs.setdefault("issues_json", "[]")
        kwargs.setdefault("correction_succeeded", 0)
        kwargs.setdefault("corrected_at", _now())
        super().__init__(**kwargs)
