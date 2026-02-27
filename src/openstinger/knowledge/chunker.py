"""
Text chunker — splits long documents into overlapping chunks.

Uses word-count based splitting (simple, tokeniser-agnostic).
For production use with tiktoken, replace with token-count splitter.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_words: int = 20,
) -> list[str]:
    """
    Split *text* into overlapping word-count chunks.

    Args:
        text:            The document text to split.
        chunk_size:      Maximum words per chunk (default 512).
        overlap:         Words shared between consecutive chunks (default 64).
        min_chunk_words: Discard trailing chunks smaller than this.

    Returns:
        List of chunk strings, each ≤ chunk_size words.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) >= min_chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride

    return chunks


def split_by_paragraphs(
    text: str,
    max_words: int = 512,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """
    Alternative: split by paragraph boundaries first, then merge short
    paragraphs into chunks up to *max_words*.

    Useful for structured documents (markdown, HTML-extracted text).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return chunk_text(text, max_words)

    chunks: list[str] = []
    current_words: list[str] = []
    current_paras: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap paragraphs
            overlap_paras = current_paras[-overlap_paragraphs:] if overlap_paragraphs else []
            current_words = " ".join(overlap_paras).split()
            current_paras = list(overlap_paras)
        current_words.extend(para_words)
        current_paras.append(para)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
