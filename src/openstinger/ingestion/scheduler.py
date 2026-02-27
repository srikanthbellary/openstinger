"""
IngestionSchedulerRegistry — manages per-agent ingestion pipelines (v0.5).

Spec: OPENSTINGER_IMPLEMENTATION_GUIDE_V3.md §IngestionSchedulerRegistry

Each named agent gets its own:
  - SessionReader instance
  - TemporalEngine instance (namespace-isolated)
  - Ingestion job tracking in operational DB

v0.5 changes:
  - Fixed episode_log bug: each ingested episode is now recorded in the
    operational DB via db.log_episode() so progress is visible in Datasette.
  - Added parallel concurrency: episodes within a batch are processed via
    asyncio.gather() controlled by the ingestion.concurrency config option.
    Default = 5 parallel episodes. Cap at 10 to stay under Novita rate limits.
    Set concurrency=1 to restore sequential behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from openstinger.ingestion.session_reader import SessionReader

logger = logging.getLogger(__name__)


class IngestionSchedulerRegistry:
    """
    Registry of active ingestion pipelines, one per agent namespace.

    Created at server startup. New agents registered via
    register_agent() (called when memory_list_agents or spawn detects a
    new namespace in session files).
    """

    def __init__(self) -> None:
        # namespace → SessionReader
        self._readers: dict[str, SessionReader] = {}
        # namespace → TemporalEngine
        self._engines: dict[str, Any] = {}
        # namespace → db_adapter (for episode_log)
        self._dbs: dict[str, Any] = {}
        # namespace → concurrency limit
        self._concurrency: dict[str, int] = {}

    async def register_agent(
        self,
        namespace: str,
        sessions_dir: Any,
        engine: Any,
        db_adapter: Any,
        poll_interval: float = 5.0,
        chunk_size: int = 10,
        session_format: str = "openclaw",
        concurrency: int = 5,
    ) -> None:
        """
        Register and start an ingestion pipeline for agent namespace.
        Idempotent — calling twice for same namespace is safe.

        Args:
            concurrency: Max number of episodes processed in parallel per batch.
                         Range 1–10. Default 5. Higher = faster but more API calls.
        """
        if namespace in self._readers:
            logger.debug("IngestionScheduler: namespace %r already registered", namespace)
            return

        self._dbs[namespace] = db_adapter
        self._concurrency[namespace] = max(1, min(concurrency, 10))

        if sessions_dir is None:
            logger.info("No sessions_dir for namespace %r — auto-ingestion disabled", namespace)
            self._engines[namespace] = engine
            return

        async def on_batch(batch: list[dict]) -> None:
            await self._process_batch(namespace, batch)

        reader = SessionReader(
            sessions_dir=sessions_dir,
            agent_namespace=namespace,
            on_batch=on_batch,
            db_adapter=db_adapter,
            poll_interval=poll_interval,
            chunk_size=chunk_size,
            session_format=session_format,
        )

        self._readers[namespace] = reader
        self._engines[namespace] = engine
        await reader.start()
        logger.info(
            "IngestionScheduler: registered namespace %r (concurrency=%d)",
            namespace, self._concurrency[namespace],
        )

    async def _process_batch(self, namespace: str, batch: list[dict]) -> None:
        """
        Process a batch of raw episode dicts from SessionReader.

        Episodes are processed in parallel (up to self._concurrency[namespace]
        at a time). Each successfully processed episode is recorded in the
        operational DB via db.log_episode() — fixing the v0.4 episode_log bug.
        """
        engine = self._engines.get(namespace)
        if engine is None:
            logger.warning("No engine for namespace %r — batch dropped", namespace)
            return

        db = self._dbs.get(namespace)
        concurrency = self._concurrency.get(namespace, 1)

        async def _ingest_one(episode_dict: dict) -> None:
            """Process a single episode and log it to the DB."""
            try:
                episode = await engine.add_episode(
                    content=episode_dict.get("content", ""),
                    source=episode_dict.get("source", "conversation"),
                    source_description=episode_dict.get("source_description", ""),
                    valid_at=episode_dict.get("valid_at"),
                    agent_namespace=namespace,
                )
                # v0.5 FIX: record episode in operational DB so progress is
                # visible in Datasette and memory_job_status can report it.
                if episode is not None and db is not None:
                    try:
                        episode_uuid = getattr(episode, "uuid", None) or str(episode)
                        await db.log_episode(
                            episode_uuid=episode_uuid,
                            agent_namespace=namespace,
                            source=episode_dict.get("source", "conversation"),
                            entity_count=0,   # TemporalEngine doesn't surface count post-call
                            edge_count=0,     # acceptable placeholder for progress tracking
                            valid_at=episode_dict.get("valid_at", int(time.time())),
                        )
                    except Exception as log_exc:
                        # Non-fatal: episode was ingested, just not logged
                        logger.debug(
                            "episode_log write failed (namespace=%r): %s", namespace, log_exc
                        )
            except Exception as exc:
                logger.error(
                    "Episode ingestion failed (namespace=%r): %s", namespace, exc
                )

        # Process episodes in parallel using a semaphore to cap concurrency
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(ep: dict) -> None:
            async with sem:
                await _ingest_one(ep)

        tasks = [asyncio.create_task(_guarded(ep)) for ep in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any unexpected exceptions that bubbled through
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Unexpected error in parallel ingestion task %d (namespace=%r): %s",
                    i, namespace, result,
                )

    async def ingest_now(self, namespace: str) -> int:
        """Trigger immediate ingestion for a namespace."""
        reader = self._readers.get(namespace)
        if reader is None:
            return 0
        return await reader.ingest_now()

    async def shutdown(self) -> None:
        """Stop all readers gracefully."""
        for namespace, reader in self._readers.items():
            await reader.stop()
            logger.info("IngestionScheduler: stopped reader for %r", namespace)
        self._readers.clear()
        self._engines.clear()
        self._dbs.clear()
        self._concurrency.clear()

    def list_namespaces(self) -> list[str]:
        return list(self._engines.keys())

    def get_engine(self, namespace: str) -> Optional[Any]:
        return self._engines.get(namespace)

    def is_registered(self, namespace: str) -> bool:
        return namespace in self._engines
