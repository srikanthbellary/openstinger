"""
StingerVault Engine — Tier 2 classification engine.

Spec: OPENSTINGER_SCAFFOLD_IMPLEMENTATION_GUIDE_V3.md

Reads episodic memory from FalkorDB on a schedule and classifies episodes
into 5 vault note categories:
  identity     — who the agent is, values, worldview
  domain       — knowledge about a subject area
  methodology  — how the agent approaches problems
  preference   — agent's preferences and style
  constraint   — hard limits and boundaries

7 classification operations:
  Extract    → pull candidate facts from episodes
  Decompose  → break compound facts into atomic notes
  Evolve     → update existing notes with new info (threshold: domain 0.85, identity 0.92)
  Link       → create semantic links between related notes
  Decay      → flag notes not confirmed in 90+ days as stale
  Organise   → generate MOC (Map of Content) files
  Log        → write classification_log entries
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

CLASSIFY_BATCH_SYSTEM = """You are an agent self-knowledge classifier.
Given a batch of episodic memory entries, extract facts that reveal persistent
truths about the agent itself — its identity, knowledge, methods, preferences, or constraints.

For each extracted note provide:
  category:    identity | domain | methodology | preference | constraint
  content:     the atomic fact statement (one clear assertion per note)
  confidence:  0.0-1.0
  related_episodes: list of episode UUIDs that support this note

Return only notes with confidence ≥ 0.6. Be conservative."""

EVOLVE_NOTE_SYSTEM = """You are a knowledge note evolution assistant.
Given an EXISTING NOTE and new evidence from recent episodes, determine:
  1. Should the note be updated? (new info adds nuance or corrects the note)
  2. If yes, provide the updated content.

Respond with JSON:
  {"should_update": bool, "updated_content": "...", "reasoning": "..."}

Be conservative: only update if the new evidence clearly changes or enriches the note.
For identity notes (category=identity), be especially conservative (threshold ≥ 0.92)."""


def _build_classify_user(episodes: list[dict]) -> str:
    episode_text = "\n\n".join(
        f"[{ep.get('uuid', 'unknown')[:8]}] {ep.get('content', '')}"
        for ep in episodes
    )
    return f"Classify agent-relevant knowledge from these episodes:\n\n{episode_text}"


def _build_evolve_user(existing_note: dict, new_episodes: list[dict]) -> str:
    ep_text = "\n".join(f"- {ep.get('content', '')}" for ep in new_episodes[:5])
    return (
        f"EXISTING NOTE (category={existing_note.get('category', '?')}):\n"
        f"{existing_note.get('content', '')}\n\n"
        f"NEW EVIDENCE:\n{ep_text}\n\n"
        f"Should this note be updated?"
    )


CLASSIFY_TOOL = {
    "name": "classify_episodes",
    "description": "Extract agent-relevant knowledge notes from episodes",
    "input_schema": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["identity", "domain", "methodology", "preference", "constraint"],
                        },
                        "content": {"type": "string"},
                        "confidence": {"type": "number"},
                        "related_episodes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["category", "content", "confidence"],
                },
            }
        },
        "required": ["notes"],
    },
}


# ---------------------------------------------------------------------------
# VaultEngine
# ---------------------------------------------------------------------------

class VaultEngine:
    """
    Reads episodic memory from FalkorDB, classifies into vault notes,
    persists notes to vault directory and knowledge graph.
    """

    def __init__(
        self,
        driver: Any,                     # FalkorDBDriver
        llm: Any,                        # AnthropicClient
        embedder: Any,                   # OpenAIEmbedder
        db: Any,                         # OperationalDBAdapter
        vault_dir: Path,
        agent_namespace: str = "default",
        episodes_per_batch: int = 20,
        domain_threshold: float = 0.85,
        identity_threshold: float = 0.92,
        decay_days: int = 90,
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.embedder = embedder
        self.db = db
        self.vault_dir = vault_dir
        self.agent_namespace = agent_namespace
        self.episodes_per_batch = episodes_per_batch
        self.domain_threshold = domain_threshold
        self.identity_threshold = identity_threshold
        self.decay_days = decay_days

        # Subdirectories
        self.self_dir = vault_dir / "self"          # identity notes
        self.notes_dir = vault_dir / "notes"        # domain/method/pref/constraint
        self.ops_dir = vault_dir / "ops"            # scratch — excluded from sync
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        moc_dir = self.vault_dir / "notes" / "moc"
        for d in [self.self_dir, self.notes_dir, self.ops_dir, moc_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main classification cycle
    # ------------------------------------------------------------------

    async def run_classification_cycle(self) -> dict:
        """
        Full classification cycle: Extract → Decompose → Evolve → Link → Decay → Organise → Log

        Returns summary stats dict.
        """
        import time as _time
        _start_ms = int(_time.time() * 1000)
        stats = {
            "episodes_processed": 0,
            "notes_created": 0,
            "notes_evolved": 0,
            "notes_decayed": 0,
            "mocs_updated": 0,
        }

        # Step 1: Fetch unclassified episodes
        episodes = await self._fetch_recent_episodes()
        stats["episodes_processed"] = len(episodes)

        if episodes:
            # Step 2: Extract + Decompose (only if there are episodes)
            raw_notes = await self._extract_notes(episodes)

            # Step 3: Evolve (merge with existing) or Create
            for note_data in raw_notes:
                created = await self._evolve_or_create_note(note_data, episodes)
                if created:
                    stats["notes_created"] += 1
                else:
                    stats["notes_evolved"] += 1
        else:
            logger.debug("VaultEngine: no new episodes to classify")

        # Step 4: Decay stale notes (always runs — independent of new episodes)
        decayed = await self._decay_stale_notes()
        stats["notes_decayed"] = decayed

        # Step 5: Organise — regenerate MOCs (always runs)
        moc_count = await self._regenerate_mocs()
        stats["mocs_updated"] = moc_count

        # Step 6: Log to operational DB
        duration_ms = int(_time.time() * 1000) - _start_ms
        await self._log_cycle(stats, duration_ms=duration_ms)

        logger.info(
            "VaultEngine cycle complete: +%d notes, %d evolved, %d decayed (%dms)",
            stats["notes_created"], stats["notes_evolved"], stats["notes_decayed"], duration_ms,
        )
        return stats

    # ------------------------------------------------------------------
    # Step 1: Fetch episodes
    # ------------------------------------------------------------------

    async def _fetch_recent_episodes(self) -> list[dict]:
        """Fetch episodes not yet classified (last N unprocessed)."""
        # Track last processed via a marker in ops/
        marker_file = self.ops_dir / ".last_classified_at"
        last_ts = 0
        if marker_file.exists():
            try:
                last_ts = int(marker_file.read_text().strip())
            except ValueError:
                last_ts = 0

        rows = await self.driver.query_temporal(
            """
            MATCH (ep:Episode {agent_namespace: $ns})
            WHERE ep.created_at > $since
            RETURN ep.uuid AS uuid, ep.content AS content,
                   ep.source AS source, ep.valid_at AS valid_at,
                   ep.created_at AS created_at
            ORDER BY ep.created_at ASC
            LIMIT $limit
            """,
            {
                "ns": self.agent_namespace,
                "since": last_ts,
                "limit": self.episodes_per_batch,
            },
        )

        if rows:
            newest_ts = max(r["created_at"] for r in rows)
            marker_file.write_text(str(newest_ts))

        return rows

    # ------------------------------------------------------------------
    # Step 2: Extract + Decompose
    # ------------------------------------------------------------------

    async def _extract_notes(self, episodes: list[dict]) -> list[dict]:
        """LLM extraction of candidate vault notes from episode batch."""
        try:
            result = await self.llm.complete_with_tools(
                system=CLASSIFY_BATCH_SYSTEM,
                user=_build_classify_user(episodes),
                tools=[CLASSIFY_TOOL],
            )
            notes = result.get("notes", [])
            # Filter by minimum confidence
            return [n for n in notes if n.get("confidence", 0) >= 0.6]
        except Exception as exc:
            logger.warning("VaultEngine: note extraction failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 3: Evolve or Create
    # ------------------------------------------------------------------

    async def _evolve_or_create_note(self, note_data: dict, episodes: list[dict]) -> bool:
        """
        Check if an existing note covers the same concept.
        If yes: evolve it (update content).
        If no: create new note.
        Returns True if newly created, False if evolved.
        """
        category = note_data.get("category", "domain")
        content = note_data.get("content", "")
        confidence = float(note_data.get("confidence", 0.85))

        # Threshold depends on category
        threshold = (
            self.identity_threshold if category == "identity" else self.domain_threshold
        )

        # Search knowledge graph for similar existing note
        existing = await self._find_similar_note(content, category)

        if existing and confidence >= threshold:
            # Evolve existing note
            await self._evolve_note(existing, note_data, episodes)
            return False
        else:
            # Create new note
            await self._create_note(note_data)
            return True

    async def _find_similar_note(
        self, content: str, category: str
    ) -> Optional[dict]:
        """Find existing note in knowledge graph via vector similarity.

        Uses cosine distance: score < 0.3 means similarity > 0.7 — close enough to evolve.
        4-arg form: (label, property, k, vecf32(query)) — required by FalkorDB 1.6+.
        """
        try:
            embedding = await self.embedder.embed(content)
            rows = await self.driver.query_knowledge(
                """
                CALL db.idx.vector.queryNodes('Note', 'content_embedding', $limit, vecf32($embedding))
                YIELD node, score
                WHERE node.agent_namespace = $ns AND node.category = $category
                      AND score < 0.3 AND node.stale = 0
                RETURN node.uuid AS uuid, node.content AS content,
                       node.category AS category, score
                LIMIT 1
                """,
                {
                    "embedding": embedding,
                    "ns": self.agent_namespace,
                    "category": category,
                    "limit": 3,
                },
            )
            return rows[0] if rows else None
        except Exception as exc:
            logger.debug("VaultEngine._find_similar_note failed: %s", exc)
            return None

    async def _evolve_note(
        self, existing: dict, new_data: dict, episodes: list[dict]
    ) -> None:
        """Update an existing note with new evidence."""
        relevant_episodes = [
            ep for ep in episodes
            if ep.get("uuid") in new_data.get("related_episodes", [])
        ]
        try:
            result = await self.llm.complete_json(
                system=EVOLVE_NOTE_SYSTEM,
                user=_build_evolve_user(existing, relevant_episodes),
            )
            if result.get("should_update") and result.get("updated_content"):
                new_content = result["updated_content"]
                await self._update_note_in_graph(existing["uuid"], new_content)
                await self._update_note_file(existing["uuid"], existing["category"], new_content)
                logger.debug("Evolved note %s", existing["uuid"][:8])
        except Exception as exc:
            logger.warning("VaultEngine: note evolution failed: %s", exc)

    async def _create_note(self, note_data: dict) -> str:
        """Create a new note in the knowledge graph and vault directory."""
        import uuid as uuidlib
        note_uuid = str(uuidlib.uuid4())
        category = note_data.get("category", "domain")
        content = note_data.get("content", "")

        try:
            embedding = await self.embedder.embed(content)
        except Exception:
            embedding = None

        now_ts = int(time.time())
        # FalkorDB Python client 1.x does not support dict as inline CREATE props.
        # Use CREATE with uuid key then SET n += $props for the rest.
        props: dict = {
            "agent_namespace": self.agent_namespace,
            "category": category,
            "content": content,
            "stale": 0,           # FalkorDB stores as int: 0=active, 1=stale
            "created_at": now_ts,
            "updated_at": now_ts,
            "last_confirmed_at": now_ts,
        }
        if embedding:
            props["content_embedding"] = embedding

        await self.driver.query_knowledge(
            "CREATE (n:Note {uuid: $uuid}) SET n += $props",
            {"uuid": note_uuid, "props": props},
        )

        # Mirror metadata to operational DB vault_notes table
        await self.db.upsert_vault_note(
            uuid=note_uuid,
            agent_namespace=self.agent_namespace,
            category=category,
            confidence=float(note_data.get("confidence", 0.85)),
        )

        # v0.6 Layer 4: link to semantically similar notes
        if embedding:
            await self._link_similar_notes(note_uuid, embedding)

        # Write markdown file
        await self._write_note_file(note_uuid, category, content)
        logger.debug("Created note %s [%s]", note_uuid[:8], category)
        return note_uuid

    async def _link_similar_notes(self, note_uuid: str, embedding: list) -> None:
        """
        After creating a note, find the top-3 semantically similar existing notes
        and create [:SIMILAR_TO] edges between them.

        Uses cosine distance < 0.25 (high similarity threshold) to avoid noise.
        Existing [:SIMILAR_TO] edges are merged (MERGE not CREATE) to stay idempotent.
        Non-fatal: if linking fails, note creation remains valid.
        """
        try:
            rows = await self.driver.query_knowledge(
                """
                CALL db.idx.vector.queryNodes('Note', 'content_embedding', $k, vecf32($emb))
                YIELD node, score
                WHERE node.agent_namespace = $ns
                  AND node.uuid <> $src_uuid
                  AND node.stale = 0
                  AND score < 0.25
                RETURN node.uuid AS target_uuid, score
                LIMIT 3
                """,
                {
                    "emb": embedding,
                    "ns": self.agent_namespace,
                    "src_uuid": note_uuid,
                    "k": 5,
                },
            )
            for row in rows:
                await self.driver.query_knowledge(
                    """
                    MATCH (a:Note {uuid: $src}), (b:Note {uuid: $tgt})
                    MERGE (a)-[:SIMILAR_TO {score: $score, created_at: $ts}]->(b)
                    """,
                    {
                        "src": note_uuid,
                        "tgt": row["target_uuid"],
                        "score": round(row["score"], 6),
                        "ts": int(time.time()),
                    },
                )
            if rows:
                logger.debug(
                    "Linked note %s to %d similar notes", note_uuid[:8], len(rows)
                )
        except Exception as exc:
            # Non-fatal: note was created, linking is best-effort
            logger.debug("_link_similar_notes failed (note=%s): %s", note_uuid[:8], exc)

    # ------------------------------------------------------------------
    # Step 4: Decay
    # ------------------------------------------------------------------

    async def _decay_stale_notes(self) -> int:
        """Flag notes not confirmed in decay_days as stale."""
        cutoff = int(time.time()) - (self.decay_days * 86400)
        rows = await self.driver.query_knowledge(
            """
            MATCH (n:Note {agent_namespace: $ns})
            WHERE n.last_confirmed_at < $cutoff AND n.stale = 0
            SET n.stale = 1
            RETURN count(n) AS count
            """,
            {"ns": self.agent_namespace, "cutoff": cutoff},
        )
        count = rows[0]["count"] if rows else 0
        if count > 0:
            logger.info("VaultEngine: decayed %d stale notes", count)
        return count

    # ------------------------------------------------------------------
    # Step 5: MOC generation
    # ------------------------------------------------------------------

    async def _regenerate_mocs(self) -> int:
        """Generate/update category MOCs and hub MOC."""
        categories = ["identity", "domain", "methodology", "preference", "constraint"]
        moc_count = 0

        for category in categories:
            notes = await self.driver.query_knowledge(
                """
                MATCH (n:Note {agent_namespace: $ns, category: $cat})
                WHERE n.stale = 0
                RETURN n.uuid AS uuid, n.content AS content
                ORDER BY n.created_at DESC
                """,
                {"ns": self.agent_namespace, "cat": category},
            )
            if notes:
                await self._write_category_moc(category, notes)
                moc_count += 1

        # Hub MOC
        await self._write_hub_moc(categories)
        moc_count += 1
        return moc_count

    # ------------------------------------------------------------------
    # Step 6: Log
    # ------------------------------------------------------------------

    async def _log_cycle(self, stats: dict, duration_ms: int | None = None) -> None:
        """Persist classification cycle log to operational DB."""
        await self.db.log_classification_cycle(
            agent_namespace=self.agent_namespace,
            episodes_processed=stats.get("episodes_processed", 0),
            notes_created=stats.get("notes_created", 0),
            notes_evolved=stats.get("notes_evolved", 0),
            notes_decayed=stats.get("notes_decayed", 0),
            mocs_updated=stats.get("mocs_updated", 0),
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _note_file_path(self, note_uuid: str, category: str) -> Path:
        if category == "identity":
            return self.self_dir / f"{note_uuid[:8]}.md"
        return self.notes_dir / f"{category}_{note_uuid[:8]}.md"

    async def _write_note_file(self, note_uuid: str, category: str, content: str) -> None:
        path = self._note_file_path(note_uuid, category)
        path.write_text(
            f"---\nuuid: {note_uuid}\ncategory: {category}\n---\n\n{content}\n",
            encoding="utf-8",
        )

    async def _update_note_file(
        self, note_uuid: str, category: str, new_content: str
    ) -> None:
        await self._write_note_file(note_uuid, category, new_content)

    async def _update_note_in_graph(self, uuid: str, new_content: str) -> None:
        now_ts = int(time.time())
        try:
            embedding = await self.embedder.embed(new_content)
        except Exception:
            embedding = None

        props: dict = {"content": new_content, "updated_at": now_ts, "last_confirmed_at": now_ts}
        if embedding:
            props["content_embedding"] = embedding

        await self.driver.query_knowledge(
            "MATCH (n:Note {uuid: $uuid}) SET n += $props",
            {"uuid": uuid, "props": props},
        )

    async def _write_category_moc(self, category: str, notes: list[dict]) -> None:
        """Write a category Map of Content file under vault/notes/moc/."""
        moc_dir = self.vault_dir / "notes" / "moc"
        moc_dir.mkdir(parents=True, exist_ok=True)
        moc_path = moc_dir / f"moc_{category}.md"
        lines = [f"# {category.upper()} — Map of Content\n"]
        for note in notes:
            snippet = note["content"][:80].replace("\n", " ")
            lines.append(f"- [[{note['uuid'][:8]}]] {snippet}")
        moc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def _write_hub_moc(self, categories: list[str]) -> None:
        """Write the hub MOC linking all category MOCs under vault/notes/moc/."""
        moc_dir = self.vault_dir / "notes" / "moc"
        moc_dir.mkdir(parents=True, exist_ok=True)
        hub_path = moc_dir / "moc_hub.md"
        lines = ["# Vault Hub — Map of Content\n"]
        for cat in categories:
            lines.append(f"- [[moc_{cat}]] — {cat}")
        hub_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public read API (used by MCP tools)
    # ------------------------------------------------------------------

    async def get_vault_stats(self) -> dict:
        """Return note counts by category."""
        categories = ["identity", "domain", "methodology", "preference", "constraint"]
        stats = {}
        for cat in categories:
            rows = await self.driver.query_knowledge(
                """
                MATCH (n:Note {agent_namespace: $ns, category: $cat})
                RETURN
                  sum(CASE WHEN n.stale = 0 THEN 1 ELSE 0 END) AS active,
                  sum(CASE WHEN n.stale = 1 THEN 1 ELSE 0 END) AS stale
                """,
                {"ns": self.agent_namespace, "cat": cat},
            )
            stats[cat] = rows[0] if rows else {"active": 0, "stale": 0}
        return stats

    async def list_notes(self, category: Optional[str] = None, include_stale: bool = False) -> dict:
        """List vault notes, optionally filtered by category.

        Returns a dict (not a bare list) so that empty results include a hint
        rather than a silent empty list — preventing LLM retry loops on empty vaults.
        """
        where_parts = ["n.agent_namespace = $ns"]
        params: dict = {"ns": self.agent_namespace}
        if category:
            where_parts.append("n.category = $cat")
            params["cat"] = category
        if not include_stale:
            where_parts.append("n.stale = 0")

        where = " AND ".join(where_parts)
        rows = await self.driver.query_knowledge(
            f"""
            MATCH (n:Note)
            WHERE {where}
            RETURN n.uuid AS uuid, n.category AS category,
                   n.content AS content, n.stale AS stale,
                   n.updated_at AS updated_at
            ORDER BY n.updated_at DESC
            """,
            params,
        )

        result: dict = {
            "category": category or "all",
            "namespace": self.agent_namespace,
            "count": len(rows),
            "notes": rows,
        }
        if not rows:
            result["hint"] = (
                f"No {'active ' if not include_stale else ''}notes found"
                + (f" in category '{category}'" if category else "")
                + ". The vault may not yet be seeded. Use memory_query to check episodic memory, "
                "or vault_note_add to manually add an identity/preference/constraint note."
            )
        return result

    async def get_note(self, uuid: str) -> Optional[dict]:
        rows = await self.driver.query_knowledge(
            "MATCH (n:Note {uuid: $uuid}) RETURN n",
            {"uuid": uuid},
        )
        return rows[0] if rows else None
