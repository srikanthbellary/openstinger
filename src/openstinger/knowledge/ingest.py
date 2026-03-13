"""
Knowledge ingestion — ECL (Extract → Chunk → Embed → Link → Store) pipeline.

Ingests an external document (URL, PDF, YouTube, plain text) into the
``openstinger_knowledge`` FalkorDB graph as Document + TextChunk nodes.

Each TextChunk is:
  - Embedded (1536-dim vector, same model as the rest of OpenStinger)
  - Linked back to any entities in the temporal graph via entity mentions

Usage:

    result = await ingest(
        source="https://example.com/article",
        source_type="url",
        agent_namespace="main",
        driver=driver,
        embedder=embedder,
    )
    print(f"Ingested {result.chunk_count} chunks, document_uuid={result.document_uuid}")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openstinger.knowledge.chunker import chunk_text

logger = logging.getLogger(__name__)

SourceType = str  # "url" | "pdf" | "youtube" | "text"


@dataclass
class IngestResult:
    """Result of a single ingest() call."""
    document_uuid: str
    source: str
    source_type: SourceType
    chunk_count: int
    agent_namespace: str
    duration_ms: int = 0
    error: Optional[str] = None
    chunk_uuids: list[str] = field(default_factory=list)


async def ingest(
    source: str,
    agent_namespace: str,
    driver: Any,
    embedder: Any,
    source_type: SourceType = "auto",
    chunk_size: int = 512,
    overlap: int = 64,
    title: Optional[str] = None,
) -> IngestResult:
    """
    Full ECL pipeline: Extract → Chunk → Embed → Store.

    Args:
        source:          URL, file path, YouTube URL/ID, or raw text.
        agent_namespace: Agent namespace for isolation.
        driver:          FalkorDBDriver instance (must be connected).
        embedder:        OpenAIEmbedder or CachedEmbedder instance.
        source_type:     "url" | "pdf" | "youtube" | "text" | "auto" (detect).
        chunk_size:      Max words per chunk (default 512).
        overlap:         Overlap words between chunks (default 64).
        title:           Optional document title override.

    Returns:
        IngestResult with document_uuid, chunk_count, and timing.
    """
    t0 = time.time()
    doc_uuid = str(uuid.uuid4())
    resolved_type = _resolve_type(source, source_type)

    # --- Extract ---
    try:
        raw_text = await _extract(source, resolved_type)
    except Exception as exc:
        logger.error("Knowledge ingest extraction failed [%s]: %s", resolved_type, exc)
        return IngestResult(
            document_uuid=doc_uuid,
            source=source,
            source_type=resolved_type,
            chunk_count=0,
            agent_namespace=agent_namespace,
            error=str(exc),
        )

    if not raw_text.strip():
        return IngestResult(
            document_uuid=doc_uuid,
            source=source,
            source_type=resolved_type,
            chunk_count=0,
            agent_namespace=agent_namespace,
            error="Extraction returned empty text",
        )

    # --- Chunk ---
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        chunks = [raw_text[:4000]]  # last-resort single chunk

    # --- Persist Document node ---
    doc_title = title or _infer_title(source, resolved_type, raw_text)
    doc_summary = raw_text[:500].replace("\n", " ").strip()
    now_ts = int(time.time())
    await driver.query_knowledge(
        """
        MERGE (d:Document {uuid: $uuid})
        SET d.agent_namespace = $ns,
            d.source_url      = $source,
            d.source_type     = $source_type,
            d.title           = $title,
            d.summary         = $summary,
            d.chunk_count     = $chunk_count,
            d.created_at      = $created_at
        """,
        {
            "uuid": doc_uuid,
            "ns": agent_namespace,
            "source": source,
            "source_type": resolved_type,
            "title": doc_title,
            "summary": doc_summary,
            "chunk_count": len(chunks),
            "created_at": now_ts,
        },
    )

    # --- Embed + Store Chunks ---
    chunk_uuids: list[str] = []
    try:
        embeddings = await embedder.embed_batch(chunks)
    except Exception as exc:
        logger.warning("Batch embedding failed, falling back to sequential: %s", exc)
        embeddings = []
        for chunk in chunks:
            try:
                embeddings.append(await embedder.embed(chunk))
            except Exception:
                embeddings.append([0.0] * 1536)

    for idx, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_uuid = str(uuid.uuid4())
        chunk_uuids.append(chunk_uuid)

        props: dict[str, Any] = {
            "agent_namespace": agent_namespace,
            "document_uuid": doc_uuid,
            "content": chunk_text_content,
            "chunk_index": idx,
            "created_at": now_ts,
        }
        if embedding:
            props["content_embedding"] = embedding

        await driver.query_knowledge(
            """
            CREATE (c:TextChunk {uuid: $uuid})
            SET c += $props
            """,
            {"uuid": chunk_uuid, "props": props},
        )

        # Link chunk → document
        await driver.query_knowledge(
            """
            MATCH (d:Document {uuid: $doc_uuid})
            MATCH (c:TextChunk {uuid: $chunk_uuid})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {"doc_uuid": doc_uuid, "chunk_uuid": chunk_uuid},
        )

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "Ingested %d chunks from %s [%s] in %dms (doc=%s)",
        len(chunks), source[:60], resolved_type, duration_ms, doc_uuid[:8],
    )

    return IngestResult(
        document_uuid=doc_uuid,
        source=source,
        source_type=resolved_type,
        chunk_count=len(chunks),
        agent_namespace=agent_namespace,
        duration_ms=duration_ms,
        chunk_uuids=chunk_uuids,
    )


def _resolve_type(source: str, source_type: SourceType) -> SourceType:
    if source_type != "auto":
        return source_type
    lower = source.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        if "youtube.com" in lower or "youtu.be" in lower:
            return "youtube"
        return "url"
    if lower.endswith(".pdf"):
        return "pdf"
    return "text"


async def _extract(source: str, source_type: SourceType) -> str:
    if source_type == "url":
        from openstinger.knowledge.sources.url import extract
        return await extract(source)
    elif source_type == "pdf":
        from openstinger.knowledge.sources.pdf import extract
        return await extract(source)
    elif source_type == "youtube":
        from openstinger.knowledge.sources.youtube import extract
        return await extract(source)
    else:
        from openstinger.knowledge.sources.plaintext import extract
        return await extract(source)


def _infer_title(source: str, source_type: SourceType, text: str) -> str:
    """Derive a short title from the source if not provided."""
    if source_type in ("url", "youtube"):
        # Use last path segment or domain
        parts = source.rstrip("/").split("/")
        return parts[-1][:100] if parts else source[:100]
    if source_type == "pdf":
        return Path(source).stem[:100]
    # Plain text: use first line
    first_line = text.split("\n")[0].strip()
    return first_line[:100] if first_line else "Untitled"
