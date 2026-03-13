"""
Three-stage entity deduplication engine.

Algorithm spec: OUTDATED_DOCS_TO_BE_RENEWED/03_ALGORITHM_REFERENCE.md §2

Stages:
  Stage 1 — Exact / normalized string match  (O(1) hash lookup)
  Stage 2 — MinHash LSH fuzzy match          (Jaccard ≥ lsh_threshold)
  Stage 3 — LLM semantic confirmation        (confidence ≥ llm_confidence_min)

Deduplication always runs BEFORE conflict resolution.
Conservative threshold (0.85) prevents false merges.
LSH index rebuilt on startup (100ms for 100k entities).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional

from datasketch import MinHash, MinHashLSH

from openstinger.temporal.nodes import EntityNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

# Corporate suffixes to strip for normalization
_CORP_SUFFIXES = re.compile(
    r"\b(inc\.?|llc\.?|ltd\.?|corp\.?|co\.?|plc\.?|gmbh|s\.a\.?|b\.v\.?)\b",
    re.IGNORECASE,
)

# Titles to strip
_TITLES = re.compile(
    r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|sir|dame)\s+",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    """
    Normalize an entity name for Stage 1 matching.
    Steps: lowercase → unicode NFKD → strip accents → strip titles →
           strip corp suffixes → collapse whitespace → strip punctuation
    """
    n = name.lower()
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = _TITLES.sub("", n)
    n = _CORP_SUFFIXES.sub("", n)
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _shingles(text: str, k: int = 3) -> set[str]:
    """Character k-shingles for MinHash."""
    text = text.replace(" ", "_")
    return {text[i: i + k] for i in range(len(text) - k + 1)} if len(text) >= k else {text}


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

DEDUP_SYSTEM = """You are an entity deduplication assistant.
Determine if the NEW ENTITY and the EXISTING ENTITY refer to the same real-world entity.

Respond with a JSON object with these keys:
  "is_same_entity": true or false
  "confidence": float 0.0-1.0
  "reasoning": one sentence

Be conservative: return is_same_entity=true only if you are highly confident.
Two different people with the same name are NOT the same entity."""


def _build_dedup_user(new_entity: EntityNode, existing_entity: dict) -> str:
    return (
        f"NEW ENTITY:\n"
        f"  name: {new_entity.name}\n"
        f"  type: {new_entity.entity_type}\n"
        f"  summary: {new_entity.summary}\n\n"
        f"EXISTING ENTITY:\n"
        f"  name: {existing_entity.get('name', '')}\n"
        f"  type: {existing_entity.get('entity_type', '')}\n"
        f"  summary: {existing_entity.get('summary', '')}\n\n"
        f"Do these refer to the same real-world entity?"
    )


# ---------------------------------------------------------------------------
# DeduplicationEngine
# ---------------------------------------------------------------------------

class DeduplicationEngine:
    """
    Three-stage entity deduplication.

    Usage:
        engine = DeduplicationEngine(llm, config)
        await engine.rebuild_lsh_index(db, namespace)
        resolved = await engine.resolve(raw_entity, namespace)
    """

    def __init__(
        self,
        llm: Any,
        lsh_threshold: float = 0.5,
        lsh_num_perm: int = 128,
        llm_confidence_min: float = 0.85,
        token_overlap_min: float = 0.4,
    ) -> None:
        self.llm = llm
        self.lsh_threshold = lsh_threshold
        self.lsh_num_perm = lsh_num_perm
        self.llm_confidence_min = llm_confidence_min
        self.token_overlap_min = token_overlap_min

        # LSH index: normalized_name → (uuid, original_name, entity_type, summary)
        self._lsh: Optional[MinHashLSH] = None
        # uuid → MinHash (for querying)
        self._minhashes: dict[str, MinHash] = {}
        # uuid → entity metadata dict
        self._entity_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # LSH index management
    # ------------------------------------------------------------------

    async def rebuild_lsh_index(
        self, driver: Any, namespace: str
    ) -> int:
        """
        Rebuild the MinHash LSH index from all entities in FalkorDB.
        Called on startup and after bulk imports.
        Returns count of entities indexed.
        """
        rows = await driver.query_temporal(
            """
            MATCH (e:Entity)
            WHERE e.agent_namespace = $namespace
            RETURN e.uuid AS uuid, e.name AS name,
                   e.entity_type AS entity_type, e.summary AS summary
            """,
            {"namespace": namespace},
        )

        self._lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.lsh_num_perm)
        self._minhashes.clear()
        self._entity_meta.clear()

        for row in rows:
            uuid = row["uuid"]
            norm = normalize_name(row["name"])
            m = self._build_minhash(norm)
            try:
                self._lsh.insert(uuid, m)
            except ValueError:
                pass  # duplicate key — already inserted
            self._minhashes[uuid] = m
            self._entity_meta[uuid] = row

        logger.info("LSH index rebuilt: %d entities (namespace=%s)", len(rows), namespace)
        return len(rows)

    def _build_minhash(self, normalized_name: str) -> MinHash:
        m = MinHash(num_perm=self.lsh_num_perm)
        for shingle in _shingles(normalized_name):
            m.update(shingle.encode("utf-8"))
        return m

    def _add_to_index(self, entity: EntityNode) -> None:
        """Add a newly registered entity to the live LSH index."""
        if self._lsh is None:
            return
        norm = normalize_name(entity.name)
        m = self._build_minhash(norm)
        try:
            self._lsh.insert(entity.uuid, m)
        except ValueError:
            pass
        self._minhashes[entity.uuid] = m
        self._entity_meta[entity.uuid] = {
            "uuid": entity.uuid,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "summary": entity.summary,
        }

    # ------------------------------------------------------------------
    # Three-stage resolution
    # ------------------------------------------------------------------

    async def resolve(
        self, new_entity: EntityNode, namespace: str
    ) -> EntityNode:
        """
        Return the canonical entity for new_entity.

        If a match is found, returns a copy of new_entity with the
        canonical uuid set.  The caller (TemporalEngine) must call
        entity_registry.get_or_register() to confirm the UUID.
        """
        norm = normalize_name(new_entity.name)

        # --- Stage 1: Exact / normalized match ---
        stage1_match = await self._stage1_exact(norm)
        if stage1_match:
            logger.debug("Dedup Stage 1 hit: '%s' → %s", new_entity.name, stage1_match[:8])
            existing = self._entity_meta.get(stage1_match, {})
            return self._merge(new_entity, stage1_match, existing)

        # --- Stage 2: LSH fuzzy match ---
        stage2_candidates = self._stage2_lsh(norm)
        if stage2_candidates:
            # Stage 3: LLM confirmation for each candidate
            for candidate_uuid in stage2_candidates:
                existing = self._entity_meta.get(candidate_uuid, {})
                confirmed, confidence = await self._stage3_llm(new_entity, existing)
                if confirmed:
                    logger.debug(
                        "Dedup Stage 3 hit: '%s' → %s (conf=%.2f)",
                        new_entity.name, candidate_uuid[:8], confidence,
                    )
                    return self._merge(new_entity, candidate_uuid, existing)

        # No match — new entity, add to index
        self._add_to_index(new_entity)
        return new_entity

    async def _stage1_exact(self, normalized: str) -> Optional[str]:
        """
        Check in-memory index for exact normalized name match.
        O(1) lookup via the _entity_meta values.
        """
        for uuid, meta in self._entity_meta.items():
            if normalize_name(meta.get("name", "")) == normalized:
                return uuid
        return None

    def _stage2_lsh(self, normalized: str) -> list[str]:
        """
        MinHash LSH approximate nearest neighbours.
        Returns list of candidate UUIDs (may be empty).
        """
        if self._lsh is None:
            return []
        m = self._build_minhash(normalized)
        try:
            return self._lsh.query(m)
        except Exception:
            return []

    async def _stage3_llm(
        self, new_entity: EntityNode, existing: dict
    ) -> tuple[bool, float]:
        """
        LLM semantic confirmation.
        Returns (is_same, confidence).
        """
        if not existing:
            return False, 0.0
        try:
            result = await self.llm.complete_json(
                system=DEDUP_SYSTEM,
                user=_build_dedup_user(new_entity, existing),
            )
            is_same = bool(result.get("is_same_entity", False))
            confidence = float(result.get("confidence", 0.0))
            if is_same and confidence >= self.llm_confidence_min:
                return True, confidence
            return False, confidence
        except Exception as exc:
            logger.warning("Dedup LLM check failed: %s", exc)
            return False, 0.0

    def cache_size(self) -> int:
        """Return number of entities currently in the in-memory index."""
        return len(self._entity_meta)

    @staticmethod
    def _merge(new_entity: EntityNode, canonical_uuid: str, existing: dict) -> EntityNode:
        """Return new_entity with canonical uuid (preserves new entity's other fields)."""
        merged = new_entity.model_copy()
        merged.uuid = canonical_uuid
        # Prefer existing summary if new one is empty
        if not merged.summary and existing.get("summary"):
            merged.summary = existing["summary"]
        return merged
