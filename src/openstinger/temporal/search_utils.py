"""BM25 / hybrid search helpers (v0.10 LME-informed)."""

from __future__ import annotations

import re

# RediSearch / FalkorDB fulltext operators and fragile tokens
_BM25_SPECIAL = re.compile(r"[\[\]\{\}\(\)\<\>\~\*\?\:\"\'\@\!\^\|\-\+\\/]")
_STOP = {
    "what", "when", "where", "who", "why", "how", "which", "whom",
    "did", "does", "do", "the", "a", "an", "i", "my", "me", "mine",
    "with", "from", "for", "to", "of", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "and", "or", "but", "if", "then",
    "that", "this", "these", "those", "you", "your", "we", "our",
    "last", "first", "not", "can", "could", "would", "should",
}


def extract_search_terms(query: str) -> list[str]:
    """Alphanumeric tokens useful for BM25 / CONTAINS, stopwords removed."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP]


def sanitize_bm25_query(query: str) -> str:
    """
    Make a natural-language question safe for FalkorDB/RediSearch BM25.

    Quotes each kept term so operators like ':' and bare 'last' do not break
    the query parser. Falls back to stripped raw text if no terms survive.
    """
    terms = extract_search_terms(query)
    if terms:
        return " ".join(f'"{t}"' for t in terms)
    cleaned = _BM25_SPECIAL.sub(" ", query or "").strip()
    # Stopword-only or empty input: avoid sending raw stopwords to BM25
    if not cleaned or not extract_search_terms(cleaned):
        return '""'
    return cleaned
