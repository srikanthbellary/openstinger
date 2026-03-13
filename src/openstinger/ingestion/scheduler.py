"""
IngestionSchedulerRegistry — manages per-agent ingestion pipelines (v0.5 / v0.6).

Spec: OPENSTINGER_IMPLEMENTATION_GUIDE_V3.md §IngestionSchedulerRegistry

Each named agent gets its own:
  - SessionReader instance
  - TemporalEngine instance (namespace-isolated)
  - Ingestion job tracking in operational DB

v0.5 changes:
  - Fixed episode_log bug: each ingested episode is now recorded in the
    operational DB via db.log_episode() so progress is visible.
  - Added parallel concurrency: episodes within a batch are processed via
    asyncio.gather() controlled by the ingestion.concurrency config option.

v0.6 changes:
  - ingestion_jobs table now populated: a job row is created before each
    batch and updated with episode count on completion.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from pathlib import Path

from openstinger.ingestion.session_reader import SessionReader
from openstinger.ingestion.profile_reader import AgentProfileIngester

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
        # namespace → AgentProfileIngester
        self._profile_readers: dict[str, AgentProfileIngester] = {}
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
        profile_dirs: list[Path] | None = None,
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
            
        if profile_dirs is not None:
            p_dirs = profile_dirs
        elif sessions_dir is not None:
            p_dirs = [Path(sessions_dir).parent]
        else:
            p_dirs = []

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
        
        profile_reader = AgentProfileIngester(
            profile_dirs=p_dirs,
            agent_namespace=namespace,
            engine=engine,
            db_adapter=db_adapter,
            poll_interval=max(60.0, poll_interval * 12),  # Poll less frequently than sessions
        )

        self._readers[namespace] = reader
        self._profile_readers[namespace] = profile_reader
        self._engines[namespace] = engine
        
        await reader.start()
        await profile_reader.start()
        
        logger.info(
            "IngestionScheduler: registered namespace %r (concurrency=%d)",
            namespace, self._concurrency[namespace],
        )

    async def _process_batch(self, namespace: str, batch: list[dict]) -> None:
        """
        Process a batch of raw episode dicts from SessionReader.

        v0.6: Creates an ingestion_jobs row before processing and updates it
        with the episode count on completion or marks it failed on error.
        Episodes are processed in parallel (up to self._concurrency[namespace]
        at a time). Each successfully processed episode is recorded in the
        operational DB via db.log_episode().
        """
        engine = self._engines.get(namespace)
        if engine is None:
            logger.warning("No engine for namespace %r — batch dropped", namespace)
            return

        db = self._dbs.get(namespace)
        concurrency = self._concurrency.get(namespace, 1)

        # v0.6: Create a job row before processing
        job = None
        if db is not None:
            try:
                # Try to get source file from first episode
                source_file = batch[0].get("session_file") if batch else None
                job = await db.create_job(
                    agent_namespace=namespace,
                    source_file=source_file,
                    source_type="session_jsonl",
                )
            except Exception as job_exc:
                logger.debug("ingestion_jobs create failed: %s", job_exc)

        successful_count = 0

        async def _ingest_one(episode_dict: dict) -> None:
            nonlocal successful_count
            """Process a single episode and log it to the DB."""
            try:
                episode = await engine.add_episode(
                    content=episode_dict.get("content", ""),
                    source=episode_dict.get("source", "conversation"),
                    source_description=episode_dict.get("source_description", ""),
                    valid_at=episode_dict.get("valid_at"),
                    agent_namespace=namespace,
                )
                # v0.5 FIX: record episode in operational DB
                if episode is not None and db is not None:
                    try:
                        episode_uuid = getattr(episode, "uuid", None) or str(episode)
                        await db.log_episode(
                            episode_uuid=episode_uuid,
                            agent_namespace=namespace,
                            source=episode_dict.get("source", "conversation"),
                            entity_count=0,
                            edge_count=0,
                            job_uuid=job.uuid if job else None,
                            valid_at=episode_dict.get("valid_at", int(time.time())),
                        )
                        successful_count += 1
                    except Exception as log_exc:
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

        # Log any unexpected exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Unexpected error in parallel ingestion task %d (namespace=%r): %s",
                    i, namespace, result,
                )

        # v0.6: Update job row with final episode count
        if job is not None and db is not None:
            try:
                job.status = "done"
                job.episodes_processed = successful_count
                await db.update_job(job)
            except Exception as job_exc:
                logger.debug("ingestion_jobs update failed: %s", job_exc)

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
