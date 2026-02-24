"""
Entity registry — cross-engine UUID coherence.

New file (not in original graphiti-core).
Ensures the same real-world entity gets the same UUID across:
  - The temporal graph (EntityEdge endpoints)
  - The knowledge graph (Note.entity_uuid references)
  - The operational DB (entity_registry table)

This prevents UUID drift when the same entity is referenced from both
episodic memory and vault notes.
"""

from __future__ import annotations

import logging
from typing import Optional

from openstinger.temporal.nodes import EntityNode

logger = logging.getLogger(__name__)


class EntityRegistry:
    """
    In-memory + DB-backed registry mapping normalized entity names to UUIDs.

    Used by:
      - TemporalEngine: register every new EntityNode
      - StingerVault: look up UUIDs when creating knowledge graph notes
    """

    def __init__(self, operational_db: "OperationalDBAdapter") -> None:  # type: ignore[name-defined]
        # Lazy import to avoid circular; typed as string
        self._db = operational_db
        # Local cache: normalized_name → uuid
        self._cache: dict[str, str] = {}

    async def warmup(self) -> None:
        """Load all known entities from DB into local cache."""
        rows = await self._db.get_all_entities()
        for row in rows:
            self._cache[row["name_normalized"]] = row["uuid"]
        logger.info("EntityRegistry warmed up: %d entities", len(self._cache))

    async def get_or_register(self, entity: EntityNode) -> str:
        """
        Return the canonical UUID for this entity.

        If the entity's normalized name is already registered, return the
        existing UUID (even if entity.uuid differs — the caller should
        update entity.uuid to match).

        If not registered, persist and return entity.uuid.
        """
        normalized = entity.name.lower().strip()

        # Fast path: cache hit
        if normalized in self._cache:
            return self._cache[normalized]

        # DB lookup (handles concurrent writers across processes)
        existing = await self._db.find_entity_by_name(normalized)
        if existing:
            self._cache[normalized] = existing["uuid"]
            return existing["uuid"]

        # New entity — persist
        await self._db.upsert_entity(
            uuid=entity.uuid,
            name=entity.name,
            name_normalized=normalized,
            entity_type=entity.entity_type,
        )
        self._cache[normalized] = entity.uuid
        logger.debug("EntityRegistry: registered new entity '%s' → %s", entity.name, entity.uuid)
        return entity.uuid

    async def touch(self, uuid: str) -> None:
        """Increment episode_count for an entity (called on each mention)."""
        await self._db.touch_entity(uuid)

    def get_cached_uuid(self, name_normalized: str) -> Optional[str]:
        """Synchronous cache lookup — returns None if not in cache."""
        return self._cache.get(name_normalized)

    def cache_size(self) -> int:
        return len(self._cache)
