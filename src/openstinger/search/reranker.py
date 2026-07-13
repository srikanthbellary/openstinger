"""Pluggable rerankers for the retrieval pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "content",
    ) -> list[dict]:
        ...


class NoopReranker:
    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "content",
    ) -> list[dict]:
        return items


class LLMReranker:
    """Score candidates with the engine LLM (fast_model when available)."""

    def __init__(self, llm: Any, *, model: str | None = None) -> None:
        self.llm = llm
        self.model = model

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "content",
    ) -> list[dict]:
        if len(items) <= 1:
            return items
        lines = []
        for i, it in enumerate(items):
            text = (it.get(text_key) or it.get("fact") or it.get("text") or "")[:400]
            lines.append(f"[{i}] {text}")
        prompt = (
            "Rank memory items by relevance to the query. "
            "Return ONLY a comma-separated list of indices from most to least relevant.\n"
            f"Query: {query}\n\nItems:\n" + "\n".join(lines)
        )
        try:
            if hasattr(self.llm, "complete"):
                raw = await self.llm.complete(
                    system=(
                        "Rank memory items by relevance. "
                        "Return ONLY a comma-separated list of indices, most relevant first."
                    ),
                    user=prompt,
                    use_fast_model=True,
                )
            else:
                return items
            text = raw if isinstance(raw, str) else str(raw)
            idxs = [int(x.strip()) for x in re_findall_indices(text) if x.strip().isdigit()]
            seen = set()
            ordered: list[dict] = []
            for i in idxs:
                if 0 <= i < len(items) and i not in seen:
                    row = dict(items[i])
                    row["rerank_score"] = float(len(items) - len(ordered))
                    ordered.append(row)
                    seen.add(i)
            for i, it in enumerate(items):
                if i not in seen:
                    ordered.append(it)
            return ordered
        except Exception as exc:
            logger.warning("LLM rerank failed, keeping RRF order: %s", exc)
            return items


def re_findall_indices(text: str) -> list[str]:
    import re
    return re.findall(r"\d+", text or "")


class CrossEncoderReranker:
    """Optional sentence-transformers cross-encoder (openstinger[rerank])."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Install openstinger[rerank] to use cross_encoder reranking"
            ) from exc
        self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "content",
    ) -> list[dict]:
        if len(items) <= 1:
            return items
        model = self._load()
        pairs = []
        for it in items:
            text = it.get(text_key) or it.get("fact") or it.get("text") or ""
            pairs.append([query, text[:2000]])

        def _score():
            return model.predict(pairs)

        scores = await asyncio.to_thread(_score)
        ranked = sorted(
            zip(items, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        out = []
        for it, sc in ranked:
            row = dict(it)
            row["rerank_score"] = float(sc)
            out.append(row)
        return out


def build_reranker(
    kind: str,
    *,
    llm: Any = None,
    timeout_ms: int = 3000,
) -> Reranker:
    kind = (kind or "none").lower()
    if kind == "cross_encoder":
        try:
            return CrossEncoderReranker()
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s); using NoopReranker", exc)
            return NoopReranker()
    if kind == "llm":
        if llm is None:
            return NoopReranker()
        return LLMReranker(llm)
    return NoopReranker()
