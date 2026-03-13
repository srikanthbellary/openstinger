"""
SessionReader — watches OpenClaw JSONL session files for new episodes.

Spec: OPENSTINGER_IMPLEMENTATION_GUIDE_V3.md §SessionReader

Supports two JSONL formats:
  "simple"   — flat dict per line: {"content":"...","source":"...","valid_at":12345}
  "openclaw" — OpenClaw rich format with typed message blocks (v3+ sessions)

OpenClaw format line types:
  session, model_change, thinking_level_change, custom  → skipped
  message (role=user|assistant, content=[text|thinking|toolCall]) → extracted

Key design decisions:
  - Byte-offset cursor: resumes from exact position, handles partial lines
  - Read-only: NEVER writes to OpenClaw session files
  - One reader per agent namespace
  - Cursors persisted in operational DB (survives restarts)
  - Poll-based (asyncio sleep), not inotify (cross-platform)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Coroutine, Literal, Optional

from openstinger.config import resolve_path

logger = logging.getLogger(__name__)

SessionFormat = Literal["simple", "openclaw"]
EpisodeBatchCallback = Callable[[list[dict]], Coroutine]


# ---------------------------------------------------------------------------
# OpenClaw JSONL format parser
# ---------------------------------------------------------------------------

def _iso_to_unix(ts: str) -> int:
    """Convert OpenClaw ISO 8601 timestamp to Unix seconds."""
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return int(datetime.now(timezone.utc).timestamp())


def parse_openclaw_line(line: str) -> Optional[dict]:
    """
    Parse one line from an OpenClaw v3 JSONL session file.

    Returns an episode dict compatible with TemporalEngine.add_episode()
    or None if this line should be skipped.

    Episode structure returned:
      content    — extracted text (user message or assistant response)
      source     — "openclaw_user" | "openclaw_assistant"
      valid_at   — unix timestamp from message timestamp
      session_id — OpenClaw session UUID (for traceability)
      message_id — OpenClaw message UUID
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Only process message lines
    if obj.get("type") != "message":
        return None

    msg = obj.get("message", {})
    role = msg.get("role", "")
    content_blocks = msg.get("content", [])
    timestamp = obj.get("timestamp", "")
    message_id = obj.get("id", "")

    # Skip messages with no role
    if role not in ("user", "assistant"):
        return None

    # Extract text blocks only (skip thinking and toolCall)
    text_parts: list[str] = []
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "").strip()
                if text:
                    text_parts.append(text)
            # Explicitly skip: thinking, toolCall, toolResult
    elif isinstance(content_blocks, str):
        # Older format: content is a plain string
        text_parts.append(content_blocks.strip())

    if not text_parts:
        return None

    content = "\n".join(text_parts)

    # Skip very short lines (unlikely to contain useful memory)
    if len(content) < 10:
        return None

    return {
        "content": content,
        "source": f"openclaw_{role}",
        "valid_at": _iso_to_unix(timestamp),
        "message_id": message_id,
    }


# ---------------------------------------------------------------------------
# SessionReader
# ---------------------------------------------------------------------------

class SessionReader:
    """
    Watches a sessions directory for OpenClaw JSONL files.

    Two format modes:
      "simple"   — each line is already a simple episode dict
      "openclaw" — OpenClaw rich JSONL (default for OpenClaw integration)

    The reader:
      1. Scans sessions_dir for *.jsonl files
      2. For each file, reads from the stored byte cursor
      3. Yields complete lines only (skips partial last line)
      4. Parses each line according to session_format
      5. Calls on_batch() callback with each chunk of episodes
      6. Advances the cursor after successful batch delivery
    """

    def __init__(
        self,
        sessions_dir: str | Path,
        agent_namespace: str,
        on_batch: EpisodeBatchCallback,
        db_adapter: "OperationalDBAdapter",  # type: ignore[name-defined]
        poll_interval: float = 5.0,
        chunk_size: int = 10,
        session_format: SessionFormat = "openclaw",
    ) -> None:
        self.sessions_dir = Path(str(resolve_path(sessions_dir)))
        self.agent_namespace = agent_namespace
        self.on_batch = on_batch
        self.db = db_adapter
        self.poll_interval = poll_interval
        self.chunk_size = chunk_size
        self.session_format: SessionFormat = session_format

        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(), name=f"session_reader_{self.agent_namespace}"
        )
        logger.info(
            "SessionReader started: dir=%s namespace=%s format=%s",
            self.sessions_dir, self.agent_namespace, self.session_format,
        )

    async def stop(self) -> None:
        """Stop background polling loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SessionReader stopped: namespace=%s", self.agent_namespace)

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._scan_and_ingest()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("SessionReader poll error: %s", exc, exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _scan_and_ingest(self) -> None:
        """Scan sessions_dir for new content and ingest."""
        if not self.sessions_dir.exists():
            logger.debug("Sessions dir not found: %s", self.sessions_dir)
            return

        jsonl_files = sorted(self.sessions_dir.glob("**/*.jsonl"))
        # Skip sessions.json (OpenClaw index file, not a session)
        jsonl_files = [f for f in jsonl_files if f.name != "sessions.json"]
        for file_path in jsonl_files:
            await self._ingest_file(file_path)

    async def _ingest_file(self, file_path: Path) -> None:
        """
        Read new lines from file_path starting from stored byte cursor.
        Partial lines at the end are skipped (cursor stays before them).
        """
        file_key = str(file_path)
        cursor = await self.db.get_cursor(self.agent_namespace, file_key)

        try:
            file_size = file_path.stat().st_size
        except OSError:
            return

        if cursor >= file_size:
            return  # No new data

        batch: list[dict] = []
        new_cursor = cursor

        # Open in binary mode for accurate byte positioning
        with open(file_path, "rb") as f:
            f.seek(cursor)
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break  # EOF

                # readline() returns a complete line (ending \n) or a
                # partial line at EOF. Skip partial lines.
                if not line_bytes.endswith(b"\n"):
                    break

                new_cursor = f.tell()
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                episode = self._parse_line(line, file_path)
                if episode is not None:
                    batch.append(episode)

                if len(batch) >= self.chunk_size:
                    await self._flush_batch(batch, file_key, new_cursor)
                    batch = []

        if batch:
            await self._flush_batch(batch, file_key, new_cursor)
        elif new_cursor > cursor:
            # Advanced cursor even if batch empty (we processed skipped lines)
            await self.db.set_cursor(self.agent_namespace, file_key, new_cursor)

    def _parse_line(self, line: str, file_path: Path) -> Optional[dict]:
        """
        Parse a single JSONL line into an episode dict, or None to skip.
        Dispatches to format-specific parser.
        """
        if self.session_format == "openclaw":
            return parse_openclaw_line(line)

        # "simple" format: each line is already an episode dict
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "content" in obj:
                return obj
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line in %s: %s",
                           str(file_path)[-40:], exc)
            return None

    async def _flush_batch(
        self, batch: list[dict], file_key: str, new_cursor: int
    ) -> None:
        """Call the callback and advance the cursor on success."""
        try:
            await self.on_batch(batch)
            await self.db.set_cursor(self.agent_namespace, file_key, new_cursor)
            logger.debug(
                "Ingested %d episodes from %s (cursor→%d)",
                len(batch), file_key[-40:], new_cursor,
            )
        except Exception as exc:
            logger.error("Batch callback failed — cursor NOT advanced: %s", exc)

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    async def ingest_now(self) -> int:
        """
        Trigger an immediate scan (used by memory_ingest_now MCP tool).
        Returns number of episodes ingested.
        """
        counter = {"count": 0}
        original_callback = self.on_batch

        async def counting_callback(batch: list[dict]) -> None:
            await original_callback(batch)
            counter["count"] += len(batch)

        self.on_batch = counting_callback
        await self._scan_and_ingest()
        self.on_batch = original_callback
        return counter["count"]
