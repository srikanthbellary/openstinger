"""
Knowledge module — ECL (Extract → Chunk → Link) pipeline for external documents.

Ingests URL, PDF, YouTube transcript, or plain text into the
openstinger_knowledge FalkorDB graph as Document + TextChunk nodes.

Entry point:
    from openstinger.knowledge.ingest import ingest
    result = await ingest(source="https://example.com", ...)
"""
from openstinger.knowledge.ingest import ingest, IngestResult

__all__ = ["ingest", "IngestResult"]
