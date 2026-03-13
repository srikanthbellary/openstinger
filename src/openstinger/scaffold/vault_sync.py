"""
Vault Sync Engine — ingests changed vault markdown files into knowledge graph.

Spec: OPENSTINGER_SCAFFOLD_IMPLEMENTATION_GUIDE_V3.md §VaultSyncEngine

Uses SHA-256 checksums to detect changed files.
Only changed files are re-ingested (efficiency at scale).
vault/ops/ is excluded from sync.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VaultSyncEngine:
    """
    Watches vault/ directory for changed markdown files and syncs them
    into the knowledge graph.

    Change detection: SHA-256 checksum stored in operational DB.
    Only files with a changed checksum are re-processed.
    """

    def __init__(
        self,
        driver: Any,
        embedder: Any,
        db: Any,
        vault_dir: Path,
        agent_namespace: str = "default",
    ) -> None:
        self.driver = driver
        self.embedder = embedder
        self.db = db
        self.vault_dir = vault_dir
        self.agent_namespace = agent_namespace
        self.ops_dir = vault_dir / "ops"

        # In-memory checksum cache: file_path → sha256
        self._checksum_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Main sync
    # ------------------------------------------------------------------

    async def sync(self) -> dict:
        """
        Scan vault/ for changed markdown files and re-ingest them.
        Returns: {"files_scanned": N, "files_synced": M, "files_unchanged": K}
        """
        import time as _time
        start_ms = int(_time.time() * 1000)
        stats = {"files_scanned": 0, "files_synced": 0, "files_unchanged": 0}

        md_files = [
            p for p in self.vault_dir.rglob("*.md")
            if not self._is_ops_path(p)
        ]
        stats["files_scanned"] = len(md_files)

        for file_path in md_files:
            changed = await self._is_changed(file_path)
            if changed:
                await self._ingest_file(file_path)
                await self._update_checksum(file_path)
                stats["files_synced"] += 1
            else:
                stats["files_unchanged"] += 1

        duration_ms = int(_time.time() * 1000) - start_ms
        await self.db.log_sync_cycle(
            agent_namespace=self.agent_namespace,
            files_scanned=stats["files_scanned"],
            files_synced=stats["files_synced"],
            files_unchanged=stats["files_unchanged"],
            duration_ms=duration_ms,
        )
        logger.info(
            "VaultSync: scanned=%d synced=%d unchanged=%d (%dms)",
            stats["files_scanned"], stats["files_synced"],
            stats["files_unchanged"], duration_ms,
        )
        return stats

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _is_ops_path(self, path: Path) -> bool:
        """Return True if path is inside the ops/ scratch directory."""
        try:
            path.relative_to(self.ops_dir)
            return True
        except ValueError:
            return False

    async def _is_changed(self, file_path: Path) -> bool:
        """Return True if file's SHA-256 differs from stored checksum."""
        current = self._sha256(file_path)
        if current is None:
            return False

        stored = self._checksum_cache.get(str(file_path))
        if stored is None:
            # Check DB
            stored = await self._load_checksum_from_db(file_path)
            if stored:
                self._checksum_cache[str(file_path)] = stored

        return current != stored

    def _sha256(self, file_path: Path) -> Optional[str]:
        try:
            return hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            return None

    async def _load_checksum_from_db(self, file_path: Path) -> Optional[str]:
        """Load stored checksum from vault_checksums table (canonical store)."""
        return await self.db.get_vault_checksum(self.agent_namespace, str(file_path))

    async def _update_checksum(self, file_path: Path) -> None:
        """Persist new checksum to vault_checksums table and update cache."""
        current = self._sha256(file_path)
        if current is None:
            return
        self._checksum_cache[str(file_path)] = current
        await self.db.set_vault_checksum(self.agent_namespace, str(file_path), current)

    # ------------------------------------------------------------------
    # File ingestion into knowledge graph
    # ------------------------------------------------------------------

    async def _ingest_file(self, file_path: Path) -> None:
        """
        Parse a vault markdown file and upsert its content into the
        knowledge graph Note node.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("VaultSync: cannot read %s: %s", file_path, exc)
            return

        note_uuid, category, content = self._parse_vault_file(file_path, text)

        try:
            embedding = await self.embedder.embed(content)
        except Exception:
            embedding = None

        now_ts = int(time.time())
        props: dict = {
            "content": content,
            "category": category,
            "agent_namespace": self.agent_namespace,
            "stale": 0,
            "updated_at": now_ts,
            "last_confirmed_at": now_ts,
        }
        if embedding:
            props["content_embedding"] = embedding

        # Upsert Note node in knowledge graph
        await self.driver.query_knowledge(
            """
            MERGE (n:Note {uuid: $uuid})
            SET n += $props
            """,
            {"uuid": note_uuid, "props": props},
        )
        logger.debug("VaultSync: synced %s → Note %s", file_path.name, note_uuid[:8])

    def _parse_vault_file(
        self, file_path: Path, text: str
    ) -> tuple[str, str, str]:
        """
        Parse a vault markdown file.

        Frontmatter format:
          ---
          uuid: <uuid>
          category: <category>
          ---
          <content>

        Falls back to file name for UUID if frontmatter missing.
        """
        import re
        import uuid as uuidlib

        uuid_val = file_path.stem.replace("_", "-")  # fallback
        category = "domain"  # fallback
        content = text

        # Parse frontmatter
        fm_match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if fm_match:
            fm_text = fm_match.group(1)
            content = fm_match.group(2).strip()

            uuid_match = re.search(r"uuid:\s*(.+)", fm_text)
            if uuid_match:
                uuid_val = uuid_match.group(1).strip()

            cat_match = re.search(r"category:\s*(.+)", fm_text)
            if cat_match:
                raw_cat = cat_match.group(1).strip()
                valid_cats = {"identity", "domain", "methodology", "preference", "constraint"}
                if raw_cat in valid_cats:
                    category = raw_cat

        # Ensure uuid_val is a valid UUID format; generate if not
        try:
            uuidlib.UUID(uuid_val)
        except ValueError:
            uuid_val = str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, str(file_path)))

        return uuid_val, category, content
