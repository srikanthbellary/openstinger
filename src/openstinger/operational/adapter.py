"""
Operational database adapter — abstract interface + SQLite/PostgreSQL implementations.

Spec: docs/05_OPERATIONAL_DB_SCHEMA.md

Switching from SQLite to PostgreSQL requires only a config change:
  operational_db:
    provider: postgresql
    postgresql_url: postgresql+asyncpg://user:pass@host/db
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from openstinger.operational.models import (
    AlignmentEvent,
    Base,
    ClassificationLog,
    CorrectionLog,
    DriftLog,
    EntityRegistryRow,
    EpisodeLog,
    IngestionJob,
    SessionState,
    SyncLog,
    VaultChecksum,
    VaultNote,
    _now,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class OperationalDBAdapter(ABC):
    """Abstract operational database interface. All 11 tables across 3 tiers."""

    @abstractmethod
    async def init(self) -> None:
        """Create tables if they don't exist."""

    @abstractmethod
    async def close(self) -> None: ...

    # -- IngestionJob --
    @abstractmethod
    async def create_job(self, agent_namespace: str, source_file: str | None, source_type: str = "session_jsonl") -> IngestionJob: ...
    @abstractmethod
    async def get_job(self, uuid: str) -> Optional[IngestionJob]: ...
    @abstractmethod
    async def update_job(self, job: IngestionJob) -> None: ...
    @abstractmethod
    async def list_jobs(self, agent_namespace: str, limit: int = 20) -> list[IngestionJob]: ...

    # -- EpisodeLog --
    @abstractmethod
    async def log_episode(self, episode_uuid: str, agent_namespace: str, source: str, entity_count: int, edge_count: int, job_uuid: str | None = None, valid_at: int | None = None) -> None: ...
    @abstractmethod
    async def get_episode_log(self, episode_uuid: str) -> Optional[EpisodeLog]: ...

    # -- EntityRegistry --
    @abstractmethod
    async def upsert_entity(self, uuid: str, name: str, name_normalized: str, entity_type: str = "ENTITY") -> None: ...
    @abstractmethod
    async def find_entity_by_name(self, name_normalized: str) -> Optional[dict]: ...
    @abstractmethod
    async def get_all_entities(self) -> list[dict]: ...
    @abstractmethod
    async def touch_entity(self, uuid: str) -> None: ...

    # -- SessionState --
    @abstractmethod
    async def get_session_state(self, agent_namespace: str) -> SessionState: ...
    @abstractmethod
    async def save_session_state(self, state: SessionState) -> None: ...
    @abstractmethod
    async def set_cursor(self, agent_namespace: str, file_path: str, byte_offset: int) -> None: ...
    @abstractmethod
    async def get_cursor(self, agent_namespace: str, file_path: str) -> int: ...

    # -- VaultNote (Tier 2) --
    @abstractmethod
    async def upsert_vault_note(self, uuid: str, agent_namespace: str, category: str, confidence: float = 0.85) -> None: ...
    @abstractmethod
    async def mark_vault_note_stale(self, uuid: str) -> None: ...
    @abstractmethod
    async def list_vault_notes(self, agent_namespace: str, category: str | None = None) -> list[VaultNote]: ...

    # -- ClassificationLog (Tier 2) --
    @abstractmethod
    async def log_classification_cycle(self, agent_namespace: str, episodes_processed: int, notes_created: int, notes_evolved: int, notes_decayed: int, mocs_updated: int, duration_ms: int | None = None) -> None: ...
    @abstractmethod
    async def get_classification_history(self, agent_namespace: str, limit: int = 20) -> list[ClassificationLog]: ...

    # -- VaultChecksum (Tier 2) --
    @abstractmethod
    async def get_vault_checksum(self, agent_namespace: str, file_path: str) -> Optional[str]: ...
    @abstractmethod
    async def set_vault_checksum(self, agent_namespace: str, file_path: str, checksum: str) -> None: ...

    # -- SyncLog (Tier 2) --
    @abstractmethod
    async def log_sync_cycle(self, agent_namespace: str, files_scanned: int, files_synced: int, files_unchanged: int, duration_ms: int | None = None) -> None: ...

    # -- AlignmentEvent (Tier 3) --
    @abstractmethod
    async def log_alignment_event(self, agent_namespace: str, verdict: str, scores: dict, issues: list[str], corrected: bool = False, profile_state: str | None = None, latency_ms: int | None = None) -> str: ...
    @abstractmethod
    async def get_alignment_events(self, agent_namespace: str, limit: int = 20) -> list[AlignmentEvent]: ...

    # -- DriftLog (Tier 3) --
    @abstractmethod
    async def log_drift_state(self, agent_namespace: str, window_size: int, mean_score: float, consecutive_flags: int, total_evaluated: int, total_flagged: int, alert_triggered: bool, window: list[float]) -> None: ...
    @abstractmethod
    async def get_drift_history(self, agent_namespace: str, limit: int = 20) -> list[DriftLog]: ...

    # -- CorrectionLog (Tier 3) --
    @abstractmethod
    async def log_correction(self, agent_namespace: str, alignment_event_uuid: str, original_text_hash: str, corrected_text_hash: str, re_eval_verdict: str | None, issues: list[str], succeeded: bool) -> None: ...


# ---------------------------------------------------------------------------
# SQLAlchemy base implementation (shared by SQLite + PostgreSQL)
# ---------------------------------------------------------------------------

class SQLAlchemyAdapter(OperationalDBAdapter):
    """
    Base implementation using SQLAlchemy async sessions.
    Subclasses provide the DSN.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._engine = create_async_engine(dsn, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Operational DB initialized: %s", self._dsn.split("///")[-1][:40])

    async def close(self) -> None:
        await self._engine.dispose()

    # -- IngestionJob --

    async def create_job(
        self,
        agent_namespace: str,
        source_file: str | None,
        source_type: str = "session_jsonl",
    ) -> IngestionJob:
        job = IngestionJob(
            agent_namespace=agent_namespace,
            source_file=source_file,
            source_type=source_type,
        )
        async with self._session_factory() as session:
            session.add(job)
            await session.commit()
            await session.refresh(job)
        return job

    async def get_job(self, uuid: str) -> Optional[IngestionJob]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(IngestionJob).where(IngestionJob.uuid == uuid)
            )
            return result.scalar_one_or_none()

    async def update_job(self, job: IngestionJob) -> None:
        async with self._session_factory() as session:
            merged = await session.merge(job)
            await session.commit()

    async def list_jobs(self, agent_namespace: str, limit: int = 20) -> list[IngestionJob]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(IngestionJob)
                .where(IngestionJob.agent_namespace == agent_namespace)
                .order_by(IngestionJob.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # -- EpisodeLog --

    async def log_episode(
        self,
        episode_uuid: str,
        agent_namespace: str,
        source: str,
        entity_count: int,
        edge_count: int,
        job_uuid: str | None = None,
        valid_at: int | None = None,
    ) -> None:
        log = EpisodeLog(
            uuid=episode_uuid,
            agent_namespace=agent_namespace,
            source=source,
            entity_count=entity_count,
            edge_count=edge_count,
            ingestion_job_uuid=job_uuid,
            valid_at=valid_at or _now(),
        )
        async with self._session_factory() as session:
            session.add(log)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                logger.debug("Episode already logged: %s", episode_uuid)

    async def get_episode_log(self, episode_uuid: str) -> Optional[EpisodeLog]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeLog).where(EpisodeLog.uuid == episode_uuid)
            )
            return result.scalar_one_or_none()

    # -- EntityRegistry --

    async def upsert_entity(
        self,
        uuid: str,
        name: str,
        name_normalized: str,
        entity_type: str = "ENTITY",
    ) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityRegistryRow).where(
                    EntityRegistryRow.name_normalized == name_normalized
                )
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                row = EntityRegistryRow(
                    uuid=uuid,
                    name=name,
                    name_normalized=name_normalized,
                    entity_type=entity_type,
                )
                session.add(row)
            else:
                existing.add_name_variant(name)
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    async def find_entity_by_name(self, name_normalized: str) -> Optional[dict]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityRegistryRow).where(
                    EntityRegistryRow.name_normalized == name_normalized
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {"uuid": row.uuid, "name": row.name, "entity_type": row.entity_type}

    async def get_all_entities(self) -> list[dict]:
        async with self._session_factory() as session:
            result = await session.execute(select(EntityRegistryRow))
            rows = result.scalars().all()
            return [
                {
                    "uuid": r.uuid,
                    "name": r.name,
                    "name_normalized": r.name_normalized,
                    "entity_type": r.entity_type,
                }
                for r in rows
            ]

    async def touch_entity(self, uuid: str) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityRegistryRow).where(EntityRegistryRow.uuid == uuid)
            )
            row = result.scalar_one_or_none()
            if row:
                row.touch()
                await session.commit()

    # -- SessionState --

    async def get_session_state(self, agent_namespace: str) -> SessionState:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SessionState).where(
                    SessionState.agent_namespace == agent_namespace
                )
            )
            state = result.scalar_one_or_none()
            if state is None:
                state = SessionState(agent_namespace=agent_namespace)
                session.add(state)
                await session.commit()
                await session.refresh(state)
            return state

    async def save_session_state(self, state: SessionState) -> None:
        async with self._session_factory() as session:
            await session.merge(state)
            await session.commit()

    async def set_cursor(
        self, agent_namespace: str, file_path: str, byte_offset: int
    ) -> None:
        state = await self.get_session_state(agent_namespace)
        state.set_cursor(file_path, byte_offset)
        await self.save_session_state(state)

    async def get_cursor(self, agent_namespace: str, file_path: str) -> int:
        state = await self.get_session_state(agent_namespace)
        cursors = state.get_cursors()
        return cursors.get(file_path, 0)

    # -- VaultNote (Tier 2) --

    async def upsert_vault_note(
        self, uuid: str, agent_namespace: str, category: str, confidence: float = 0.85
    ) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(VaultNote).where(VaultNote.uuid == uuid)
            )
            row = result.scalar_one_or_none()
            now = _now()
            if row is None:
                row = VaultNote(
                    uuid=uuid, agent_namespace=agent_namespace,
                    category=category, confidence=confidence,
                    created_at=now, updated_at=now, last_confirmed_at=now,
                )
                session.add(row)
            else:
                row.category = category
                row.confidence = confidence
                row.updated_at = now
                row.last_confirmed_at = now
                row.stale = 0
            await session.commit()

    async def mark_vault_note_stale(self, uuid: str) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(VaultNote).where(VaultNote.uuid == uuid)
            )
            row = result.scalar_one_or_none()
            if row:
                row.stale = 1
                await session.commit()

    async def list_vault_notes(
        self, agent_namespace: str, category: str | None = None
    ) -> list[VaultNote]:
        async with self._session_factory() as session:
            q = select(VaultNote).where(
                VaultNote.agent_namespace == agent_namespace,
                VaultNote.stale == 0,
            )
            if category:
                q = q.where(VaultNote.category == category)
            result = await session.execute(q.order_by(VaultNote.updated_at.desc()))
            return list(result.scalars().all())

    # -- ClassificationLog (Tier 2) --

    async def log_classification_cycle(
        self, agent_namespace: str, episodes_processed: int, notes_created: int,
        notes_evolved: int, notes_decayed: int, mocs_updated: int,
        duration_ms: int | None = None,
    ) -> None:
        async with self._session_factory() as session:
            row = ClassificationLog(
                agent_namespace=agent_namespace,
                episodes_processed=episodes_processed,
                notes_created=notes_created,
                notes_evolved=notes_evolved,
                notes_decayed=notes_decayed,
                mocs_updated=mocs_updated,
                duration_ms=duration_ms,
            )
            session.add(row)
            await session.commit()

    async def get_classification_history(
        self, agent_namespace: str, limit: int = 20
    ) -> list[ClassificationLog]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ClassificationLog)
                .where(ClassificationLog.agent_namespace == agent_namespace)
                .order_by(ClassificationLog.completed_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # -- VaultChecksum (Tier 2) --

    async def get_vault_checksum(
        self, agent_namespace: str, file_path: str
    ) -> Optional[str]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(VaultChecksum).where(
                    VaultChecksum.agent_namespace == agent_namespace,
                    VaultChecksum.file_path == file_path,
                )
            )
            row = result.scalar_one_or_none()
            return row.checksum_sha256 if row else None

    async def set_vault_checksum(
        self, agent_namespace: str, file_path: str, checksum: str
    ) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(VaultChecksum).where(
                    VaultChecksum.agent_namespace == agent_namespace,
                    VaultChecksum.file_path == file_path,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = VaultChecksum(
                    agent_namespace=agent_namespace,
                    file_path=file_path,
                    checksum_sha256=checksum,
                    last_synced_at=_now(),
                )
                session.add(row)
            else:
                row.checksum_sha256 = checksum
                row.last_synced_at = _now()
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    # -- SyncLog (Tier 2) --

    async def log_sync_cycle(
        self, agent_namespace: str, files_scanned: int, files_synced: int,
        files_unchanged: int, duration_ms: int | None = None,
    ) -> None:
        async with self._session_factory() as session:
            row = SyncLog(
                agent_namespace=agent_namespace,
                files_scanned=files_scanned,
                files_synced=files_synced,
                files_unchanged=files_unchanged,
                duration_ms=duration_ms,
            )
            session.add(row)
            await session.commit()

    # -- AlignmentEvent (Tier 3) --

    async def log_alignment_event(
        self, agent_namespace: str, verdict: str, scores: dict, issues: list[str],
        corrected: bool = False, profile_state: str | None = None,
        latency_ms: int | None = None,
    ) -> str:
        import json as _json
        event_uuid = __import__("uuid").uuid4().__str__()
        async with self._session_factory() as session:
            row = AlignmentEvent(
                uuid=event_uuid,
                agent_namespace=agent_namespace,
                verdict=verdict,
                value_coherence_score=scores.get("value_coherence"),
                identity_consistent=(
                    1 if scores.get("identity_consistent") is True
                    else 0 if scores.get("identity_consistent") is False
                    else None
                ),
                constraint_compliant=(
                    1 if scores.get("constraint_compliant") is True
                    else 0 if scores.get("constraint_compliant") is False
                    else None
                ),
                content_safe=(
                    1 if scores.get("content_safe") is True
                    else 0 if scores.get("content_safe") is False
                    else None
                ),
                issues_json=_json.dumps(issues),
                scores_json=_json.dumps({k: v for k, v in scores.items()
                                         if isinstance(v, (int, float, bool, str))}),
                corrected=1 if corrected else 0,
                profile_state=profile_state,
                latency_ms=latency_ms,
            )
            session.add(row)
            await session.commit()
        return event_uuid

    async def get_alignment_events(
        self, agent_namespace: str, limit: int = 20
    ) -> list[AlignmentEvent]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(AlignmentEvent)
                .where(AlignmentEvent.agent_namespace == agent_namespace)
                .order_by(AlignmentEvent.evaluated_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # -- DriftLog (Tier 3) --

    async def log_drift_state(
        self, agent_namespace: str, window_size: int, mean_score: float,
        consecutive_flags: int, total_evaluated: int, total_flagged: int,
        alert_triggered: bool, window: list[float],
    ) -> None:
        import json as _json
        soft_flag_rate = total_flagged / total_evaluated if total_evaluated > 0 else 0.0
        async with self._session_factory() as session:
            row = DriftLog(
                agent_namespace=agent_namespace,
                window_size=window_size,
                mean_score=round(mean_score, 4),
                consecutive_flags=consecutive_flags,
                total_evaluated=total_evaluated,
                total_flagged=total_flagged,
                soft_flag_rate=round(soft_flag_rate, 4),
                alert_triggered=1 if alert_triggered else 0,
                window_json=_json.dumps(window),
            )
            session.add(row)
            await session.commit()

    async def get_drift_history(
        self, agent_namespace: str, limit: int = 20
    ) -> list[DriftLog]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DriftLog)
                .where(DriftLog.agent_namespace == agent_namespace)
                .order_by(DriftLog.recorded_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    # -- CorrectionLog (Tier 3) --

    async def log_correction(
        self, agent_namespace: str, alignment_event_uuid: str,
        original_text_hash: str, corrected_text_hash: str,
        re_eval_verdict: str | None, issues: list[str], succeeded: bool,
    ) -> None:
        import json as _json
        async with self._session_factory() as session:
            row = CorrectionLog(
                agent_namespace=agent_namespace,
                alignment_event_uuid=alignment_event_uuid,
                original_text_hash=original_text_hash,
                corrected_text_hash=corrected_text_hash,
                re_eval_verdict=re_eval_verdict,
                issues_json=_json.dumps(issues),
                correction_succeeded=1 if succeeded else 0,
            )
            session.add(row)
            await session.commit()


# ---------------------------------------------------------------------------
# SQLite adapter
# ---------------------------------------------------------------------------

class SQLiteAdapter(SQLAlchemyAdapter):
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dsn = f"sqlite+aiosqlite:///{path}"
        super().__init__(dsn)


# ---------------------------------------------------------------------------
# PostgreSQL adapter
# ---------------------------------------------------------------------------

class PostgreSQLAdapter(SQLAlchemyAdapter):
    def __init__(self, dsn: str) -> None:
        # Ensure asyncpg driver
        if "postgresql://" in dsn and "asyncpg" not in dsn:
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://")
        super().__init__(dsn)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(
    provider: str,
    sqlite_path: str | Path | None = None,
    postgresql_url: str | None = None,
) -> OperationalDBAdapter:
    if provider == "sqlite":
        if not sqlite_path:
            raise ValueError("sqlite_path required for SQLite provider")
        return SQLiteAdapter(sqlite_path)
    elif provider == "postgresql":
        if not postgresql_url:
            raise ValueError("postgresql_url required for PostgreSQL provider")
        return PostgreSQLAdapter(postgresql_url)
    else:
        raise ValueError(f"Unknown operational DB provider: {provider!r}")
