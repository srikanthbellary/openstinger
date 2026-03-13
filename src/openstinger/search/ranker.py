"""
Hybrid search score normalisation and result merging.

FalkorDB score conventions:
  - BM25 (fulltext): unbounded positive integer — higher = more relevant
  - Vector (cosine): cosine *distance* in [0, 2] — lower = more similar

Both are converted to a unified relevance score in [0.0, 1.0] where
higher always means more relevant. Results can then be merged and sorted.
"""

from __future__ import annotations


def normalize_bm25(rows: list[dict], score_key: str = "score") -> list[dict]:
    """
    Min-max normalise BM25 scores from unbounded integers to [0.0, 1.0].

    If all scores are equal (or there is only one result) every item gets 1.0.
    The original score is preserved as ``bm25_raw``.
    """
    if not rows:
        return rows
    scores = [float(r.get(score_key, 0)) for r in rows]
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [
            {**r, score_key: 1.0, "bm25_raw": float(r.get(score_key, 0))}
            for r in rows
        ]
    return [
        {
            **r,
            "bm25_raw": float(r.get(score_key, 0)),
            score_key: round((float(r.get(score_key, 0)) - mn) / (mx - mn), 4),
        }
        for r in rows
    ]


def dist_to_similarity(rows: list[dict], score_key: str = "score") -> list[dict]:
    """
    Convert cosine distance [0, 2] → cosine similarity [0.0, 1.0].

    similarity = max(0, 1 - distance)

    The original distance is preserved as ``vector_dist``.
    """
    return [
        {
            **r,
            "vector_dist": float(r.get(score_key, 1.0)),
            score_key: round(max(0.0, 1.0 - float(r.get(score_key, 1.0))), 4),
        }
        for r in rows
    ]


def merge_and_rank(
    *result_lists: list[dict],
    id_key: str = "uuid",
    score_key: str = "score",
    limit: int = 20,
) -> list[dict]:
    """
    Merge results from multiple search legs, deduplicate by ``id_key``, and
    return a single list sorted by ``score_key`` descending.

    For duplicates (same uuid appearing in multiple lists) the highest score
    is kept. Assumes all scores have already been normalised to [0.0, 1.0].
    """
    merged: dict[str, dict] = {}
    for rows in result_lists:
        for item in rows:
            uid = item.get(id_key)
            if uid is None:
                continue
            existing = merged.get(uid)
            item_score = float(item.get(score_key, 0.0))
            if existing is None or item_score > float(existing.get(score_key, 0.0)):
                merged[uid] = item

    return sorted(
        merged.values(),
        key=lambda x: float(x.get(score_key, 0.0)),
        reverse=True,
    )[:limit]
