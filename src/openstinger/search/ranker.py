"""
Hybrid search score normalisation, RRF fusion, and confidence signals.

FalkorDB score conventions:
  - BM25 (fulltext): unbounded positive integer — higher = more relevant
  - Vector (cosine): cosine distance in [0, 2] — lower = more similar

Display helpers still normalise to [0.0, 1.0]. Ranking uses Reciprocal Rank Fusion.
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
    Convert cosine distance [0, 2] to cosine similarity [0.0, 1.0].

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
    Legacy max-score merge (pre-RRF). Prefer ``rrf_fuse`` for ranking.
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


def rrf_fuse(
    channel_results: dict[str, list[dict]],
    weights: dict[str, float] | None = None,
    k: int = 60,
    id_key: str = "uuid",
    limit: int = 20,
) -> list[dict]:
    """
    Reciprocal Rank Fusion across named channels.

    score(d) = sum over channels: weight_c / (k + rank_c(d)).
    Missing channel membership contributes nothing.
    Tie-break: higher fusion score, then uuid ascending.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    channels_hit: dict[str, set[str]] = {}

    for channel, rows in channel_results.items():
        if not rows:
            continue
        w = float(weights.get(channel, 1.0))
        for rank, row in enumerate(rows, start=1):
            uid = row.get(id_key)
            if not uid:
                continue
            scores[uid] = scores.get(uid, 0.0) + w / (k + rank)
            channels_hit.setdefault(uid, set()).add(channel)
            prev = payloads.get(uid)
            if prev is None:
                payloads[uid] = {**row}
            else:
                # Keep richest payload fields
                merged = {**prev, **{kk: vv for kk, vv in row.items() if vv is not None}}
                payloads[uid] = merged

    ordered = sorted(
        scores.keys(),
        key=lambda uid: (-scores[uid], str(uid)),
    )[:limit]

    out: list[dict] = []
    for uid in ordered:
        row = dict(payloads[uid])
        row["fusion_score"] = round(scores[uid], 6)
        row["score"] = row["fusion_score"]
        row["rrf_channels"] = sorted(channels_hit.get(uid, set()))
        out.append(row)
    return out


def retrieval_confidence(
    fused: list[dict],
    *,
    abstain_threshold: float = 0.15,
) -> tuple[float, bool]:
    """
    Bounded confidence from top fusion score and multi-channel agreement.

    Returns (confidence in [0, 1], abstain_suggested).
    """
    if not fused:
        return 0.0, True
    top = fused[0]
    raw = float(top.get("fusion_score") or top.get("score") or 0.0)
    # Soft saturate: RRF tops are often << 1
    conf = min(1.0, raw * 4.0)
    channels = top.get("rrf_channels") or []
    if len(channels) >= 2:
        conf = min(1.0, conf + 0.2)
    conf = round(conf, 4)
    return conf, conf < abstain_threshold
