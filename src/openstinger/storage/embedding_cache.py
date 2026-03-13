"""
SQLite-backed embedding cache.

Cache key = SHA-256(text + "|" + model_name).
Cache hit  = zero API cost, sub-millisecond latency.
Cache miss = call upstream embedder, store result.

Usage:

    cache = EmbeddingCache(db_path=Path(".openstinger/embed_cache.db"), model_name="text-embedding-3-small")
    await cache.init()

    # Direct use
    vec = await cache.get("hello world")
    if vec is None:
        vec = await upstream_embedder.embed("hello world")
        await cache.put("hello world", vec)

    # Decorator pattern — wrap an OpenAIEmbedder
    cached_embedder = CachedEmbedder(embedder=my_embedder, cache=cache)
    vec = await cached_embedder.embed("hello world")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    SQLite-backed embedding cache.

    Thread-safe (aiosqlite uses a dedicated thread).
    Creates the cache table on first init().
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        cache_key   TEXT PRIMARY KEY,
        model_name  TEXT NOT NULL,
        embedding   TEXT NOT NULL,      -- JSON array of floats
        hit_count   INTEGER DEFAULT 0,
        created_at  INTEGER NOT NULL,
        last_hit_at INTEGER NOT NULL
    )
    """
    _CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_embed_model ON embedding_cache (model_name)"

    def __init__(self, db_path: Path, model_name: str) -> None:
        self._db_path = db_path
        self._model_name = model_name
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        """Create the cache table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(self._CREATE_TABLE)
            await db.execute(self._CREATE_INDEX)
            await db.commit()
        logger.debug("EmbeddingCache initialised: %s", self._db_path)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{text}|{self._model_name}".encode()).hexdigest()

    async def get(self, text: str) -> Optional[list[float]]:
        """Return cached embedding or None on cache miss."""
        key = self._cache_key(text)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT embedding FROM embedding_cache WHERE cache_key = ?", (key,)
                ) as cursor:
                    row = await cursor.fetchone()
                if row:
                    # Update hit stats asynchronously (best-effort)
                    now = int(time.time())
                    await db.execute(
                        "UPDATE embedding_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                        "WHERE cache_key = ?",
                        (now, key),
                    )
                    await db.commit()
                    return json.loads(row[0])
        except Exception as exc:
            logger.debug("EmbeddingCache.get error: %s", exc)
        return None

    async def put(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        key = self._cache_key(text)
        now = int(time.time())
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache
                        (cache_key, model_name, embedding, hit_count, created_at, last_hit_at)
                    VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (key, self._model_name, json.dumps(embedding), now, now),
                )
                await db.commit()
        except Exception as exc:
            logger.debug("EmbeddingCache.put error: %s", exc)

    async def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT COUNT(*), SUM(hit_count) FROM embedding_cache WHERE model_name = ?",
                    (self._model_name,),
                ) as cursor:
                    row = await cursor.fetchone()
            total_entries = row[0] if row else 0
            total_hits = row[1] if row and row[1] else 0
            return {
                "model_name": self._model_name,
                "total_entries": total_entries,
                "total_hits": int(total_hits),
                "db_path": str(self._db_path),
            }
        except Exception:
            return {"model_name": self._model_name, "total_entries": 0, "total_hits": 0}

    async def clear(self) -> int:
        """Delete all cache entries for this model. Returns deleted count."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM embedding_cache WHERE model_name = ?", (self._model_name,)
                )
                await db.commit()
                return cursor.rowcount or 0
        except Exception:
            return 0


class CachedEmbedder:
    """
    Wraps any embedder with a read-through embedding cache.

    Drop-in replacement for OpenAIEmbedder when a cache is available.

    Example:
        base = OpenAIEmbedder(api_key=..., model=...)
        cache = EmbeddingCache(Path(".openstinger/embed_cache.db"), model)
        await cache.init()
        embedder = CachedEmbedder(base, cache)
        # Use embedder exactly like OpenAIEmbedder
    """

    def __init__(self, embedder: Any, cache: EmbeddingCache) -> None:
        self._embedder = embedder
        self._cache = cache

    async def embed(self, text: str) -> list[float]:
        cached = await self._cache.get(text)
        if cached is not None:
            return cached
        embedding = await self._embedder.embed(text)
        await self._cache.put(text, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        cached_map: dict[int, list[float]] = {}
        for i, text in enumerate(texts):
            vec = await self._cache.get(text)
            if vec is not None:
                cached_map[i] = vec
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch-fetch uncached texts
        if uncached_texts:
            new_embeddings = await self._embedder.embed_batch(uncached_texts)
            for idx, (text, emb) in zip(uncached_indices, zip(uncached_texts, new_embeddings)):
                cached_map[idx] = emb
                await self._cache.put(text, emb)

        return [cached_map[i] for i in range(len(texts))]
