"""
IngestionSchedulerRegistry — manages per-agent ingestion pipelines (v0.3).

Spec: OPENSTINGER_IMPLEMENTATION_GUIDE_V3.md §IngestionSchedulerRegistry

Each named agent gets its own:
  - SessionReader instance
  - TemporalEngine instance (namespace-isolated)
  - Ingestion job tracking in operational DB
"""

from __future__ import annotations

import logging
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

    async def register_agent(
        self,
        namespace: str,
        sessions_dir: Any,
        engine: Any,
        db_adapter: Any,
        poll_interval: float = 5.0,
        chunk_size: int = 10,
        session_format: str = "openclaw",
    ) -> None:
        """
        Register and start an ingestion pipeline for agent namespace.
        Idempotent — calling twice for same namespace is safe.
        """
        if namespace in self._readers:
            logger.debug("IngestionScheduler: namespace %r already registered", namespace)
            return

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
        logger.info("IngestionScheduler: registered namespace %r", namespace)

    async def _process_batch(self, namespace: str, batch: list[dict]) -> None:
        """Process a batch of raw episode dicts from SessionReader."""
        engine = self._engines.get(namespace)
        if engine is None:
            logger.warning("No engine for namespace %r — batch dropped", namespace)
            return

        for episode_dict in batch:
            try:
                await engine.add_episode(
                    content=episode_dict.get("content", ""),
                    source=episode_dict.get("source", "conversation"),
                    source_description=episode_dict.get("source_description", ""),
                    valid_at=episode_dict.get("valid_at"),
                    agent_namespace=namespace,
                )
            except Exception as exc:
                logger.error(
                    "Episode ingestion failed (namespace=%r): %s", namespace, exc
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

    def list_namespaces(self) -> list[str]:
        return list(self._engines.keys())

    def get_engine(self, namespace: str) -> Optional[Any]:
        return self._engines.get(namespace)

    def is_registered(self, namespace: str) -> bool:
        return namespace in self._engines
