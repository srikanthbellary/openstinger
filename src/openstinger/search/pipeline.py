"""
Unified retrieval pipeline (v0.10 wave 2).

Channels → RRF → optional rerank → C1/C2 packaging → optional digests.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openstinger.config import RetrievalConfig
from openstinger.search.packer import pack_items
from openstinger.search.ranker import rrf_fuse, retrieval_confidence
from openstinger.search.reranker import build_reranker
from openstinger.temporal.search_utils import (
    apply_preference_boost,
    apply_preference_context_boost,
    apply_query_noun_boost,
    apply_recency_packaging,
    diversify_by_source,
    effective_search_limit,
    extract_query_focus_terms,
    extract_search_terms,
    extract_subqueries,
    extract_topic_nouns,
    extract_topic_phrases,
    force_include_expertise_episodes,
    force_include_focus_episodes,
    force_include_topic_episodes,
    is_activity_duration_query,
    is_count_query,
    is_errand_count_query,
    is_preference_context_query,
    is_recommend_query,
    is_topic_inventory_query,
    package_episode_row,
    prioritize_query_entity_recency,
    sanitize_bm25_query,
    build_activity_duration_digest,
    build_expertise_digest,
    build_inventory_digest,
    build_preference_digest,
    build_topic_inventory_digest,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {
    "bm25_primary": 1.0,
    "bm25_subqueries": 0.9,
    "vector_episodes": 1.0,
    "contains_terms": 0.7,
    "focus_contains": 0.6,
    "graph_episodes": 0.8,
    "graph_facts": 0.8,
    "vector_facts": 0.9,
    "vector_statements": 1.0,
    "bm25_statements": 1.0,
    "recency": 0.5,
}


class RetrievalPipeline:
    """Owns channel execution, RRF fusion, rerank, and post-fusion packaging."""

    def __init__(self, engine: Any, config: RetrievalConfig | None = None) -> None:
        self.engine = engine
        self.config = config or RetrievalConfig()
        self._reranker = None

    def _weights(self) -> dict[str, float]:
        w = dict(_DEFAULT_WEIGHTS)
        w.update(self.config.channel_weights or {})
        return w

    def _fetch_limit(self, limit: int, query: str = "") -> int:
        eff = effective_search_limit(query, limit) if query else max(limit, 1)
        raw = eff * max(self.config.candidate_multiplier, 1)
        return min(raw, self.config.candidate_cap)

    async def search(
        self,
        query: str,
        *,
        agent_namespace: str | None = None,
        limit: int = 10,
        include_expired: bool = False,
        after_unix: int | None = None,
        before_unix: int | None = None,
        max_tokens: int | None = None,
        candidate_multiplier: int | None = None,
    ) -> dict[str, Any]:
        engine = self.engine
        namespace = agent_namespace or engine.agent_namespace
        cfg = self.config
        if candidate_multiplier is not None:
            # temporary override for reflect
            old_m = cfg.candidate_multiplier
            cfg.candidate_multiplier = candidate_multiplier
            try:
                return await self._search_inner(
                    query,
                    namespace=namespace,
                    limit=limit,
                    include_expired=include_expired,
                    after_unix=after_unix,
                    before_unix=before_unix,
                    max_tokens=max_tokens,
                )
            finally:
                cfg.candidate_multiplier = old_m
        return await self._search_inner(
            query,
            namespace=namespace,
            limit=limit,
            include_expired=include_expired,
            after_unix=after_unix,
            before_unix=before_unix,
            max_tokens=max_tokens,
        )

    async def _search_inner(
        self,
        query: str,
        *,
        namespace: str,
        limit: int,
        include_expired: bool,
        after_unix: int | None,
        before_unix: int | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        engine = self.engine
        cfg = self.config
        fetch_limit = self._fetch_limit(limit, query)
        bm25_query = sanitize_bm25_query(query)
        terms = extract_search_terms(query)
        for phrase in extract_topic_phrases(query):
            if phrase not in terms:
                terms.append(phrase)
        for noun in extract_topic_nouns(query):
            if noun not in terms:
                terms.append(noun)
        subqueries = extract_subqueries(
            query, experimental_lexicons=cfg.experimental_lexicons
        )
        query_embedding = await engine.embedder.embed(query)

        time_filter = ""
        time_params: dict[str, Any] = {}
        if after_unix is not None:
            time_filter += " AND node.valid_at >= $after_unix"
            time_params["after_unix"] = after_unix
        if before_unix is not None:
            time_filter += " AND node.valid_at <= $before_unix"
            time_params["before_unix"] = before_unix

        ep_return = (
            "node.uuid AS uuid, node.content AS content, "
            "node.valid_at AS valid_at, node.valid_at_human AS valid_at_human, "
            "node.source_description AS source_description, score"
        )

        channels: dict[str, list[dict]] = {}
        channels_run: list[str] = []

        async def _bm25(q: str) -> list[dict]:
            try:
                return await engine.driver.query_temporal(
                    f"""
                    CALL db.idx.fulltext.queryNodes('Episode', $query)
                    YIELD node, score
                    WHERE node.agent_namespace = $namespace{time_filter}
                    RETURN {ep_return}
                    ORDER BY score DESC LIMIT $limit
                    """,
                    {
                        "query": sanitize_bm25_query(q),
                        "namespace": namespace,
                        "limit": fetch_limit,
                        **time_params,
                    },
                )
            except Exception as exc:
                logger.warning("Episode BM25 failed (%r): %s", q, exc)
                return []

        # Channel: bm25_primary
        primary = await _bm25(query)
        channels["bm25_primary"] = primary
        channels_run.append("bm25_primary")

        # Channel: bm25_subqueries (each subquery as soft agreement via merge ranks)
        sq_rows: list[dict] = []
        seen_sq: set[str] = set()
        for sq in subqueries[1:]:
            for r in await _bm25(sq):
                uid = r.get("uuid")
                if uid and uid not in seen_sq:
                    seen_sq.add(uid)
                    sq_rows.append(r)
        if sq_rows:
            channels["bm25_subqueries"] = sq_rows
            channels_run.append("bm25_subqueries")

        # Channel: vector_episodes
        try:
            vec_rows = await engine.driver.query_temporal(
                f"""
                CALL db.idx.vector.queryNodes('Episode', 'content_embedding', $limit, vecf32($embedding))
                YIELD node, score
                WHERE node.agent_namespace = $namespace{time_filter}
                RETURN {ep_return}
                """,
                {
                    "embedding": query_embedding,
                    "namespace": namespace,
                    "limit": fetch_limit,
                    **time_params,
                },
            )
            channels["vector_episodes"] = vec_rows
            channels_run.append("vector_episodes")
        except Exception as exc:
            logger.warning("Episode vector search unavailable: %s", exc)

        # Channel: contains_terms
        contains_rows = await self._contains_channel(
            namespace, terms, subqueries, fetch_limit, after_unix, before_unix, time_params
        )
        if contains_rows:
            channels["contains_terms"] = contains_rows
            channels_run.append("contains_terms")

        # Focus terms → generic CONTAINS (no score injection); entity routing in graph
        focus_terms = extract_query_focus_terms(query)
        if cfg.experimental_lexicons:
            focus_terms = self._experimental_focus_extras(query, focus_terms)
        focus_rows = await self._focus_contains(
            namespace, focus_terms, fetch_limit, after_unix, before_unix, time_params
        )
        if focus_rows:
            channels["focus_contains"] = focus_rows
            channels_run.append("focus_contains")

        # Entities + graph expansion
        entity_rows = await self._vector_entities(namespace, query_embedding, limit)
        entity_uuids = await self._seed_entity_uuids(
            namespace, focus_terms, entity_rows, query_embedding
        )
        graph_ep, graph_fa = await self._graph_expand(
            namespace, entity_uuids, fetch_limit, after_unix, before_unix, time_params, include_expired
        )
        if graph_ep:
            channels["graph_episodes"] = graph_ep
            channels_run.append("graph_episodes")
        if graph_fa:
            channels["graph_facts"] = graph_fa
            channels_run.append("graph_facts")

        # Facts vector
        fact_rows = await self._vector_facts(
            namespace, query_embedding, fetch_limit, include_expired
        )
        if fact_rows:
            channels["vector_facts"] = fact_rows
            channels_run.append("vector_facts")

        # Statements (may be empty before backfill)
        st_vec, st_bm25 = await self._statement_channels(
            namespace, query, query_embedding, fetch_limit, time_filter, time_params
        )
        if st_vec:
            channels["vector_statements"] = st_vec
            channels_run.append("vector_statements")
        if st_bm25:
            channels["bm25_statements"] = st_bm25
            channels_run.append("bm25_statements")

        # Optional recency channel for current-state queries
        if self._looks_current(query):
            rec = await self._recency_channel(
                namespace, fetch_limit, after_unix, before_unix, time_params
            )
            if rec:
                channels["recency"] = rec
                channels_run.append("recency")

        # Experimental lexicon expertise pass (quarantined)
        if cfg.experimental_lexicons:
            exp_rows = await self._experimental_expertise_pass(
                query, namespace, channels, fetch_limit, after_unix, before_unix, time_params
            )
            if exp_rows:
                channels["experimental_expertise"] = exp_rows
                channels_run.append("experimental_expertise")

        # Fuse episode-like channels separately from facts/entities for packaging
        episode_channels = {
            k: v
            for k, v in channels.items()
            if k
            not in {
                "vector_facts",
                "graph_facts",
                "vector_statements",
                "bm25_statements",
            }
            or k.startswith("bm25_statements")
            or k.startswith("vector_statements")
        }
        # Statements are episode-like for ranking into episode list when they carry episode uuid
        # For now statements returned as separate facts-like; also promote to episode if linked
        fused_pool = rrf_fuse(
            {k: v for k, v in channels.items() if k not in {"vector_facts", "graph_facts"}},
            weights=self._weights(),
            k=cfg.rrf_k,
            limit=fetch_limit,
        )

        # Rerank (preserve order; append unscored tail)
        reranker_name = cfg.reranker
        reranker = build_reranker(
            reranker_name,
            llm=getattr(engine, "llm", None),
            timeout_ms=cfg.rerank_timeout_ms,
        )
        head = fused_pool[: cfg.rerank_top_n]
        tail = fused_pool[cfg.rerank_top_n :]
        try:
            import asyncio

            reranked = await asyncio.wait_for(
                reranker.rerank(query, head, text_key="content"),
                timeout=cfg.rerank_timeout_ms / 1000.0,
            )
            # Only rewrite scores when a real reranker ran (noop must keep RRF scores)
            if reranker_name and reranker_name != "none":
                for i, row in enumerate(reranked):
                    row["fusion_score"] = float(len(reranked) - i)
                    if row.get("rerank_score") is not None:
                        row["score"] = float(row["rerank_score"])
                    else:
                        row["score"] = row["fusion_score"]
            fused_pool = reranked + tail
        except Exception as exc:
            logger.debug("Rerank skipped/failed: %s", exc)

        # Promote statement hits to parent episodes when linked
        fused_pool = await self._hydrate_statement_hits(fused_pool)

        # Selective post-fusion boosts (avoid reordering errand counts: RRF already
        # retrieves clothing gold; answer digests fix aggregation).
        if is_recommend_query(query):
            fused_pool = apply_preference_boost(fused_pool, query)
            fused_pool = force_include_expertise_episodes(fused_pool, query, fetch_limit)
        if (
            is_preference_context_query(query)
            or is_topic_inventory_query(query)
            or is_activity_duration_query(query)
        ):
            fused_pool = apply_query_noun_boost(fused_pool, query)
            fused_pool = apply_preference_context_boost(fused_pool, query)
            phrases = extract_topic_phrases(query)
            # Activity questions often use single tokens (jogging, yoga)
            act_terms = extract_topic_nouns(query) if is_activity_duration_query(query) else []
            match_terms = list(dict.fromkeys(phrases + act_terms))
            if match_terms:
                by_uuid = {r.get("uuid"): r for r in fused_pool if r.get("uuid")}
                for r in contains_rows or []:
                    uid = r.get("uuid")
                    if not uid:
                        continue
                    cl = (r.get("content") or "").lower()
                    if not any(t in cl for t in match_terms):
                        continue
                    if uid in by_uuid:
                        if not by_uuid[uid].get("content") and r.get("content"):
                            by_uuid[uid]["content"] = r.get("content")
                    else:
                        by_uuid[uid] = dict(r)
                fused_pool = list(by_uuid.values())
            fused_pool = force_include_topic_episodes(
                fused_pool, query, fetch_limit
            )
            # Keep boosted ranks ahead of stale RRF fusion_score values
            for r in fused_pool:
                if r.get("topic_forced") or r.get("noun_boost") or r.get("preference_boost"):
                    r["fusion_score"] = float(r.get("score") or 0)
        fused_pool = force_include_focus_episodes(
            fused_pool, focus_rows or [], fetch_limit
        )

        boosted = is_recommend_query(query) or is_preference_context_query(query)

        # C1 labels/conflicts
        labeled, conflicts = apply_recency_packaging(fused_pool)
        if boosted or is_topic_inventory_query(query) or is_activity_duration_query(query):
            ordered = sorted(
                labeled,
                key=lambda r: (
                    0 if r.get("topic_forced") or r.get("expertise_forced") else 1,
                    -float(r.get("score") or 0),
                ),
            )
        else:
            # Preserve RRF/rerank order for fair errand / factoid paths
            order_ids = [r.get("uuid") for r in fused_pool if r.get("uuid")]
            by_id = {r.get("uuid"): r for r in labeled if r.get("uuid")}
            ordered = [by_id[uid] for uid in order_ids if uid in by_id]
            for r in labeled:
                uid = r.get("uuid")
                if uid and uid not in by_id:
                    ordered.append(r)

        # C2 diversity then package
        diversified = diversify_by_source(ordered, fetch_limit)
        episodes = [
            package_episode_row(r, query)
            for r in diversified
            if r.get("content") is not None or r.get("result_type") == "episode"
        ]
        episodes = prioritize_query_entity_recency(episodes, query)
        episodes = episodes[:limit]

        if max_tokens is not None:
            episodes, tokens_used, items_dropped = pack_items(
                episodes, max_tokens=max_tokens, text_key="content"
            )
        else:
            tokens_used = 0
            items_dropped = 0

        # Entity / fact lists (RRF among themselves lightly)
        entity_sim = self._dist_to_sim(entity_rows)
        fact_merged = rrf_fuse(
            {
                k: channels[k]
                for k in ("vector_facts", "graph_facts")
                if k in channels
            },
            weights=self._weights(),
            k=cfg.rrf_k,
            limit=limit,
        ) or self._dist_to_sim(fact_rows)

        conf, abstain = retrieval_confidence(
            fused_pool, abstain_threshold=cfg.abstain_threshold
        )

        digests: dict[str, str] = {}
        if cfg.keep_digests:
            if is_recommend_query(query):
                pref = build_preference_digest(episodes)
                if pref:
                    digests["preferences"] = pref
                exp = build_expertise_digest(episodes, query)
                if exp:
                    digests["expertise"] = exp
            if is_errand_count_query(query):
                inv = build_inventory_digest(episodes, query)
                if inv:
                    digests["inventory"] = inv
            elif is_activity_duration_query(query):
                act = build_activity_duration_digest(episodes, query)
                if act:
                    digests["activity"] = act
            elif is_count_query(query) or is_topic_inventory_query(query):
                top = build_topic_inventory_digest(episodes, query)
                if top:
                    digests["topic_inventory"] = top

        ranked = (
            [{**r, "result_type": "episode"} for r in episodes]
            + [{**r, "result_type": "entity"} for r in entity_sim[:limit]]
            + [{**r, "result_type": "fact"} for r in fact_merged[:limit]]
        )
        ranked = sorted(
            ranked,
            key=lambda x: float(x.get("fusion_score") or x.get("score") or 0),
            reverse=True,
        )[:limit]

        return {
            "episodes": episodes,
            "entities": entity_sim[:limit],
            "facts": fact_merged[:limit],
            "ranked": ranked,
            "conflicts": conflicts,
            "bm25_query": bm25_query,
            "subqueries": subqueries,
            "digests": digests,
            "retrieval_confidence": conf,
            "abstain_suggested": abstain,
            "tokens_used": tokens_used,
            "items_dropped": items_dropped,
            "pipeline": {
                "channels_run": channels_run,
                "reranker": reranker_name,
                "rrf_k": cfg.rrf_k,
                "experimental_lexicons": cfg.experimental_lexicons,
            },
        }

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    async def _contains_channel(
        self, namespace, terms, subqueries, fetch_limit, after_unix, before_unix, time_params
    ):
        engine = self.engine
        contain_terms = list(
            dict.fromkeys(terms + [t for sq in subqueries[1:] for t in extract_search_terms(sq)])
        )
        if not contain_terms:
            return []
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for kw in sorted(set(contain_terms), key=lambda w: (-len(w), w))[:12]:
            try:
                # Multi-word phrases need a wide scan: CONTAINS has no relevance
                # order. Do not ORDER BY time (that drops older answer sessions).
                kw_limit = 150 if " " in kw else max(fetch_limit, 40)
                rows = await engine.driver.query_temporal(
                    f"""
                    MATCH (ep:Episode {{agent_namespace: $namespace}})
                    WHERE toLower(ep.content) CONTAINS $kw
                    {"AND ep.valid_at >= $after_unix" if after_unix is not None else ""}
                    {"AND ep.valid_at <= $before_unix" if before_unix is not None else ""}
                    RETURN ep.uuid AS uuid, ep.content AS content,
                           ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                           ep.source_description AS source_description
                    LIMIT $limit
                    """,
                    {
                        "namespace": namespace,
                        "kw": kw.lower(),
                        "limit": kw_limit,
                        **time_params,
                    },
                )
                for r in rows:
                    uid = r.get("uuid")
                    if not uid:
                        continue
                    # Prefer first-person ownership language for inventory phrases
                    cl = (r.get("content") or "").lower()
                    owned = 0.35 if (
                        " " in kw
                        and re.search(r"\b(?:my|i(?:'ve| have)?)\b.{0,60}" + re.escape(kw), cl)
                    ) else 0.0
                    scores[uid] = scores.get(uid, 0.0) + 1.0 + (len(kw) / 20.0) + owned
                    payloads[uid] = {**r, "score": scores[uid], "search_type": "contains"}
            except Exception as exc:
                logger.debug("CONTAINS failed for %r: %s", kw, exc)
        return sorted(payloads.values(), key=lambda r: r.get("score", 0), reverse=True)[
            :fetch_limit
        ]

    async def _focus_contains(
        self, namespace, focus_terms, fetch_limit, after_unix, before_unix, time_params
    ):
        engine = self.engine
        out: list[dict] = []
        seen: set[str] = set()
        for focus in focus_terms[:8]:
            try:
                rows = await engine.driver.query_temporal(
                    f"""
                    MATCH (ep:Episode {{agent_namespace: $namespace}})
                    WHERE toLower(ep.content) CONTAINS $kw
                    {"AND ep.valid_at >= $after_unix" if after_unix is not None else ""}
                    {"AND ep.valid_at <= $before_unix" if before_unix is not None else ""}
                    RETURN ep.uuid AS uuid, ep.content AS content,
                           ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                           ep.source_description AS source_description
                    ORDER BY ep.valid_at DESC
                    LIMIT $limit
                    """,
                    {
                        "namespace": namespace,
                        "kw": focus.lower(),
                        "limit": max(fetch_limit, 20),
                        **time_params,
                    },
                )
                for r in rows:
                    uid = r.get("uuid")
                    if uid and uid not in seen:
                        seen.add(uid)
                        out.append({**r, "search_type": "focus_contains", "focus_term": focus})
            except Exception as exc:
                logger.debug("Focus CONTAINS failed for %r: %s", focus, exc)
        return out

    async def _vector_entities(self, namespace, embedding, limit):
        try:
            return await self.engine.driver.query_temporal(
                """
                CALL db.idx.vector.queryNodes('Entity', 'name_embedding', $limit, vecf32($embedding))
                YIELD node, score
                WHERE node.agent_namespace = $namespace
                RETURN node.uuid AS uuid, node.name AS name,
                       node.entity_type AS entity_type, score
                """,
                {"embedding": embedding, "namespace": namespace, "limit": max(limit, 5)},
            )
        except Exception as exc:
            logger.debug("Entity vector failed: %s", exc)
            return []

    async def _seed_entity_uuids(self, namespace, focus_terms, entity_rows, embedding):
        uuids: list[str] = []
        for r in entity_rows[:5]:
            if r.get("uuid"):
                uuids.append(r["uuid"])
        # Exact name match for focus terms
        for term in focus_terms[:6]:
            try:
                rows = await self.engine.driver.query_temporal(
                    """
                    MATCH (e:Entity {agent_namespace: $namespace})
                    WHERE toLower(e.name) = $name OR toLower(e.name) CONTAINS $name
                    RETURN e.uuid AS uuid LIMIT 3
                    """,
                    {"namespace": namespace, "name": term.lower()},
                )
                for r in rows:
                    if r.get("uuid") and r["uuid"] not in uuids:
                        uuids.append(r["uuid"])
            except Exception:
                continue
        return uuids[:8]

    async def _graph_expand(
        self, namespace, entity_uuids, fetch_limit, after_unix, before_unix, time_params, include_expired
    ):
        if not entity_uuids:
            return [], []
        engine = self.engine
        ep_rows: list[dict] = []
        try:
            ep_rows = await engine.driver.query_temporal(
                f"""
                MATCH (ep:Episode {{agent_namespace: $namespace}})-[:MENTIONS]->(e:Entity)
                WHERE e.uuid IN $entity_uuids
                {"AND ep.valid_at >= $after_unix" if after_unix is not None else ""}
                {"AND ep.valid_at <= $before_unix" if before_unix is not None else ""}
                RETURN DISTINCT ep.uuid AS uuid, ep.content AS content,
                       ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                       ep.source_description AS source_description
                ORDER BY ep.valid_at DESC
                LIMIT $limit
                """,
                {
                    "namespace": namespace,
                    "entity_uuids": entity_uuids,
                    "limit": fetch_limit,
                    **time_params,
                },
            )
        except Exception as exc:
            logger.debug("Graph episode expand failed: %s", exc)

        fact_rows: list[dict] = []
        expired = "" if include_expired else "AND r.expired_at IS NULL"
        try:
            fact_rows = await engine.driver.query_temporal(
                f"""
                MATCH (e:Entity)-[r:RELATES_TO]-(:Entity)
                WHERE e.uuid IN $entity_uuids AND r.agent_namespace = $namespace {expired}
                RETURN r.uuid AS uuid, r.fact AS fact, r.valid_from AS valid_from,
                       r.relation_type AS relation_type
                LIMIT $limit
                """,
                {
                    "namespace": namespace,
                    "entity_uuids": entity_uuids,
                    "limit": fetch_limit,
                },
            )
        except Exception as exc:
            logger.debug("Graph fact expand failed: %s", exc)
        return ep_rows, fact_rows

    async def _vector_facts(self, namespace, embedding, limit, include_expired):
        expired = "" if include_expired else "AND r.expired_at IS NULL"
        try:
            return await self.engine.driver.query_temporal(
                f"""
                CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', $limit, vecf32($embedding))
                YIELD relationship AS r, score
                WHERE r.agent_namespace = $namespace {expired}
                RETURN r.uuid AS uuid, r.fact AS fact,
                       r.relation_type AS relation_type,
                       r.valid_from AS valid_from, r.expired_at AS expired_at, score
                """,
                {"embedding": embedding, "namespace": namespace, "limit": limit},
            )
        except Exception as exc:
            logger.debug("Fact vector failed: %s", exc)
            return []

    async def _statement_channels(
        self, namespace, query, embedding, fetch_limit, time_filter, time_params
    ):
        engine = self.engine
        vec: list[dict] = []
        bm25: list[dict] = []
        try:
            vec = await engine.driver.query_temporal(
                f"""
                CALL db.idx.vector.queryNodes('Statement', 'text_embedding', $limit, vecf32($embedding))
                YIELD node, score
                WHERE node.agent_namespace = $namespace
                RETURN node.uuid AS uuid, node.text AS text, node.text AS content,
                       node.kind AS kind, node.valid_at AS valid_at,
                       node.source_description AS source_description,
                       node.episode_uuid AS episode_uuid, score
                """,
                {"embedding": embedding, "namespace": namespace, "limit": fetch_limit},
            )
        except Exception as exc:
            logger.debug("Statement vector unavailable: %s", exc)
        try:
            bm25 = await engine.driver.query_temporal(
                f"""
                CALL db.idx.fulltext.queryNodes('Statement', $query)
                YIELD node, score
                WHERE node.agent_namespace = $namespace
                RETURN node.uuid AS uuid, node.text AS text, node.text AS content,
                       node.kind AS kind, node.valid_at AS valid_at,
                       node.source_description AS source_description,
                       node.episode_uuid AS episode_uuid, score
                ORDER BY score DESC LIMIT $limit
                """,
                {
                    "query": sanitize_bm25_query(query),
                    "namespace": namespace,
                    "limit": fetch_limit,
                },
            )
        except Exception as exc:
            logger.debug("Statement BM25 unavailable: %s", exc)
        return vec, bm25

    async def _hydrate_statement_hits(self, rows: list[dict]) -> list[dict]:
        """
        Promote Statement hits to parent Episode content when DISTILLED_TO exists.

        Keeps statement text as via_statement / cue so ranking still reflects the atom,
        while answer packaging gets full episode context and session attribution.
        """
        if not rows:
            return rows
        stmt_ids = [
            r.get("uuid")
            for r in rows
            if r.get("uuid") and (r.get("episode_uuid") or r.get("kind"))
        ]
        if not stmt_ids:
            for row in rows:
                if row.get("result_type") is None:
                    if row.get("content") is not None:
                        row["result_type"] = "episode"
                    elif row.get("text") is not None:
                        row["result_type"] = "statement"
                        row.setdefault("content", row.get("text"))
            return rows

        by_stmt: dict[str, dict] = {}
        try:
            linked = await self.engine.driver.query_temporal(
                """
                UNWIND $uuids AS suid
                MATCH (ep:Episode)-[:DISTILLED_TO]->(s:Statement {uuid: suid})
                RETURN s.uuid AS stmt_uuid, s.text AS stmt_text, s.kind AS kind,
                       ep.uuid AS uuid, ep.content AS content,
                       ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                       ep.source_description AS source_description
                """,
                {"uuids": list(dict.fromkeys(stmt_ids))},
            )
            for row in linked or []:
                suid = row.get("stmt_uuid")
                if suid:
                    by_stmt[suid] = row
        except Exception as exc:
            logger.debug("Statement hydrate failed: %s", exc)

        out: list[dict] = []
        seen_eps: set[str] = set()
        for row in rows:
            suid = row.get("uuid")
            parent = by_stmt.get(suid) if suid else None
            if parent and parent.get("uuid"):
                ep_uuid = parent["uuid"]
                if ep_uuid in seen_eps:
                    continue
                seen_eps.add(ep_uuid)
                merged = dict(parent)
                merged["result_type"] = "episode"
                merged["via_statement"] = parent.get("stmt_text") or row.get("text")
                merged["kind"] = parent.get("kind") or row.get("kind")
                merged["score"] = row.get("score")
                merged["fusion_score"] = row.get("fusion_score", row.get("score"))
                merged["rerank_score"] = row.get("rerank_score")
                if merged.get("via_statement"):
                    merged["cue_summary"] = f"retained: {merged['via_statement'][:200]}"
                out.append(merged)
                continue
            if row.get("result_type") is None:
                if row.get("content") is not None and row.get("kind") is None:
                    row["result_type"] = "episode"
                elif row.get("text") is not None:
                    row["result_type"] = "statement"
                    row.setdefault("content", row.get("text"))
            # Deduplicate episodes already promoted
            uid = row.get("uuid")
            if row.get("result_type") == "episode" and uid and uid in seen_eps:
                continue
            if row.get("result_type") == "episode" and uid:
                seen_eps.add(uid)
            out.append(row)
        return out

    async def _recency_channel(self, namespace, fetch_limit, after_unix, before_unix, time_params):
        try:
            return await self.engine.driver.query_temporal(
                f"""
                MATCH (ep:Episode {{agent_namespace: $namespace}})
                WHERE true
                {"AND ep.valid_at >= $after_unix" if after_unix is not None else ""}
                {"AND ep.valid_at <= $before_unix" if before_unix is not None else ""}
                RETURN ep.uuid AS uuid, ep.content AS content,
                       ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                       ep.source_description AS source_description
                ORDER BY ep.valid_at DESC
                LIMIT $limit
                """,
                {"namespace": namespace, "limit": fetch_limit, **time_params},
            )
        except Exception as exc:
            logger.debug("Recency channel failed: %s", exc)
            return []

    def _looks_current(self, query: str) -> bool:
        q = (query or "").lower()
        return any(w in q for w in ("current", "now", "latest", "still", "these days"))

    def _experimental_focus_extras(self, query: str, focus_terms: list[str]) -> list[str]:
        from openstinger.search import experimental_lexicons as xl
        out = list(focus_terms)
        ql = query.lower()
        if is_recommend_query(query) and "hotel" in ql:
            for extra in xl.HOTEL_FOCUS_EXTRAS:
                if extra not in [t.lower() for t in out]:
                    out.append(extra)
        if is_recommend_query(query) and any(
            w in ql for w in ("publication", "conference", "paper", "journal", "interesting")
        ):
            for extra in xl.PUB_FOCUS_EXTRAS:
                if extra not in [t.lower() for t in out]:
                    out.append(extra)
        return out

    async def _experimental_expertise_pass(
        self, query, namespace, channels, fetch_limit, after_unix, before_unix, time_params
    ):
        from openstinger.search import experimental_lexicons as xl
        from openstinger.temporal.search_utils import score_domain_depth

        if not is_recommend_query(query):
            return []
        ql = query.lower()
        if not any(w in ql for w in ("publication", "conference", "paper", "journal", "interesting")):
            return []
        # Scan existing channel payloads for domain depth
        domain_scores: dict[str, float] = {}
        for rows in channels.values():
            for row in rows:
                content = row.get("content") or ""
                if xl.SOP_NOISE_RE.search(content):
                    continue
                domain, depth = score_domain_depth(content)
                if domain and depth >= 3.0:
                    domain_scores[domain] = max(domain_scores.get(domain, 0), depth)
        if not domain_scores:
            return []
        top_domain = max(domain_scores, key=domain_scores.get)
        out: list[dict] = []
        seen: set[str] = set()
        for kw in list(xl.DOMAIN_LEXICONS.get(top_domain, ()))[:4]:
            try:
                rows = await self.engine.driver.query_temporal(
                    f"""
                    MATCH (ep:Episode {{agent_namespace: $namespace}})
                    WHERE toLower(ep.content) CONTAINS $kw
                    {"AND ep.valid_at >= $after_unix" if after_unix is not None else ""}
                    {"AND ep.valid_at <= $before_unix" if before_unix is not None else ""}
                    RETURN ep.uuid AS uuid, ep.content AS content,
                           ep.valid_at AS valid_at, ep.valid_at_human AS valid_at_human,
                           ep.source_description AS source_description
                    ORDER BY ep.valid_at DESC LIMIT $limit
                    """,
                    {
                        "namespace": namespace,
                        "kw": kw.lower(),
                        "limit": fetch_limit,
                        **time_params,
                    },
                )
                for r in rows:
                    uid = r.get("uuid")
                    if uid and uid not in seen:
                        seen.add(uid)
                        out.append({**r, "expertise_domain": top_domain})
            except Exception:
                continue
        return out

    @staticmethod
    def _dist_to_sim(rows: list[dict]) -> list[dict]:
        return [
            {
                **r,
                "score": round(max(0.0, 1.0 - float(r.get("score", 1.0))), 4),
                "search_type": "vector",
            }
            for r in rows
        ]
