"""
Tier 2 (Scaffold) MCP server.

Routes all 9 Tier 1 tools unchanged + adds 11 new Scaffold tools = 20 total.

New tools:
  vault_status      — vault health and last sync time
  vault_sync_now    — trigger immediate vault file sync
  vault_stats       — note counts by category
  vault_promote_now — trigger immediate classification cycle
  vault_note_list   — list notes (filtered by category)
  vault_note_get    — fetch a single note by UUID
  vault_note_add    — manually create a vault note (v0.7)
  knowledge_ingest  — ingest URL / PDF / YouTube / text into knowledge graph
  namespace_list    — list all registered agent namespaces
  namespace_create  — create a new named agent namespace
  namespace_archive — soft-archive an agent namespace
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from openstinger.config import load_config
from openstinger.mcp.server import OpenStingerServer
from openstinger.mcp.tools.memory_tools import (
    memory_add, memory_query, memory_search,
    memory_get_entity, memory_get_episode, memory_job_status,
    memory_ingest_now, memory_namespace_status, memory_list_agents,
)
from openstinger.scaffold.vault_engine import VaultEngine
from openstinger.scaffold.vault_sync import VaultSyncEngine

# APScheduler is optional — fall back to raw asyncio loops if not installed
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler as _APScheduler
    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False
    _APScheduler = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Tier 1 tools (pass-through)
from openstinger.mcp.server import TOOL_SCHEMAS as TIER1_TOOLS

# Tier 2 tool schemas
TIER2_TOOLS = [
    types.Tool(
        name="vault_status",
        description="Get vault health, last sync time, and note counts.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="vault_sync_now",
        description="Trigger immediate sync of changed vault markdown files into knowledge graph.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="vault_stats",
        description="Get note counts by category (active/stale).",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="vault_promote_now",
        description="Trigger immediate StingerVault classification cycle.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="vault_note_list",
        description="List vault notes, optionally filtered by category.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["identity", "domain", "methodology", "preference", "constraint"],
                },
                "include_stale": {"type": "boolean", "default": False},
            },
        },
    ),
    types.Tool(
        name="vault_note_add",
        description=(
            "Manually create a vault note in a specific category. "
            "Use this to seed agent identity, preferences, constraints, or domain knowledge "
            "before the automatic classification cycle has run. "
            "Categories: identity | domain | methodology | preference | constraint."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["identity", "domain", "methodology", "preference", "constraint"],
                    "description": "Vault note category",
                },
                "content": {
                    "type": "string",
                    "description": "Note content — a clear, declarative statement about the agent",
                },
                "confidence": {
                    "type": "number",
                    "default": 0.90,
                    "description": "Confidence score 0.0–1.0 (manual notes default to 0.90)",
                },
            },
            "required": ["category", "content"],
        },
    ),
    types.Tool(
        name="vault_note_get",
        description="Fetch a single vault note by UUID.",
        inputSchema={
            "type": "object",
            "properties": {"uuid": {"type": "string"}},
            "required": ["uuid"],
        },
    ),
    types.Tool(
        name="knowledge_ingest",
        description=(
            "Ingest an external document (URL, PDF, YouTube video, or plain text) "
            "into the knowledge graph as searchable chunks."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "URL, file path, YouTube URL/ID, or raw text content",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["url", "pdf", "youtube", "text", "auto"],
                    "default": "auto",
                    "description": "Source type — 'auto' detects from source string",
                },
                "title": {
                    "type": "string",
                    "description": "Optional document title override",
                },
            },
            "required": ["source"],
        },
    ),
    types.Tool(
        name="namespace_list",
        description="List all registered agent namespaces.",
        inputSchema={
            "type": "object",
            "properties": {
                "include_archived": {"type": "boolean", "default": False},
            },
        },
    ),
    types.Tool(
        name="namespace_create",
        description="Create a new named agent namespace with its own temporal graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable agent name (e.g. 'research-agent')",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="namespace_archive",
        description="Archive a named agent namespace (soft-delete, data preserved).",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "UUID of the agent to archive"},
            },
            "required": ["agent_id"],
        },
    ),
]

ALL_TOOLS = TIER1_TOOLS + TIER2_TOOLS


class ScaffoldServer:
    """Tier 2 MCP server wrapping Tier 1 + adding vault + knowledge + namespace tools."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mcp = Server("openstinger-scaffold")
        self.tier1: Any = None        # OpenStingerServer
        self.vault_engine: Any = None
        self.vault_sync: Any = None
        self._last_sync_at: int = 0
        self._scheduler: Any = None   # APScheduler or None
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.mcp.list_tools()
        async def list_tools() -> list[types.Tool]:
            return ALL_TOOLS

        @self.mcp.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            result = await self._dispatch(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _dispatch(self, name: str, args: dict) -> Any:
        # Tier 1 pass-through
        if name.startswith("memory_"):
            return await self.tier1._dispatch(name, args)

        # Tier 2 tools
        match name:
            case "vault_status":
                return await self._vault_status()
            case "vault_sync_now":
                return await self._vault_sync_now()
            case "vault_stats":
                return await self.vault_engine.get_vault_stats()
            case "vault_promote_now":
                return await self.vault_engine.run_classification_cycle()
            case "vault_note_list":
                return await self.vault_engine.list_notes(**args)
            case "vault_note_get":
                note = await self.vault_engine.get_note(args["uuid"])
                return note or {"found": False}
            case "vault_note_add":
                note_data = {
                    "category": args["category"],
                    "content": args["content"],
                    "confidence": args.get("confidence", 0.90),
                }
                note_uuid = await self.vault_engine._create_note(note_data)
                return {
                    "success": True,
                    "uuid": note_uuid,
                    "category": args["category"],
                    "message": f"Vault note created in category '{args['category']}'",
                }
            case "knowledge_ingest":
                return await self._knowledge_ingest(
                    source=args["source"],
                    source_type=args.get("source_type", "auto"),
                    title=args.get("title"),
                )
            case "namespace_list":
                return await self._namespace_list(
                    include_archived=args.get("include_archived", False)
                )
            case "namespace_create":
                return await self._namespace_create(name=args["name"])
            case "namespace_archive":
                return await self._namespace_archive(agent_id=args["agent_id"])
            case _:
                return {"error": f"Unknown tool: {name}"}

    async def _vault_status(self) -> dict:
        import time
        stats = await self.vault_engine.get_vault_stats()
        total = sum(v.get("active", 0) for v in stats.values())
        return {
            "status": "healthy",
            "namespace": self.cfg.agent_namespace,
            "total_active_notes": total,
            "last_sync_at": self._last_sync_at,
            "vault_dir": str(self.cfg.resolved_vault_dir()),
        }

    async def _vault_sync_now(self) -> dict:
        import time
        result = await self.vault_sync.sync()
        self._last_sync_at = int(time.time())
        return result

    async def startup(self) -> None:
        """Start Tier 1 then initialise Tier 2 components."""
        cfg = self.cfg

        # Tier 1
        self.tier1 = OpenStingerServer(cfg)
        await self.tier1.startup()

        vault_dir = cfg.resolved_vault_dir()

        self.vault_engine = VaultEngine(
            driver=self.tier1.driver,
            llm=self.tier1.llm,
            embedder=self.tier1.embedder,
            db=self.tier1.db,
            vault_dir=vault_dir,
            agent_namespace=cfg.agent_namespace,
            episodes_per_batch=cfg.vault.episodes_per_classification_batch,
            domain_threshold=cfg.deduplication.llm_confidence_min,
            identity_threshold=cfg.deduplication.identity_confidence_min,
            decay_days=cfg.vault.decay_days,
        )

        self.vault_sync = VaultSyncEngine(
            driver=self.tier1.driver,
            embedder=self.tier1.embedder,
            db=self.tier1.db,
            vault_dir=vault_dir,
            agent_namespace=cfg.agent_namespace,
        )

        # Start background scheduling (APScheduler if available, else raw asyncio loops)
        self._start_scheduler()

        logger.info(
            "Scaffold server ready: namespace=%s scheduler=%s",
            cfg.agent_namespace,
            "apscheduler" if _HAS_APSCHEDULER else "asyncio",
        )

    def _start_scheduler(self) -> None:
        """Start background classification + sync jobs."""
        classify_interval = self.cfg.vault.classification_interval_seconds
        sync_interval = self.cfg.vault.sync_interval_seconds

        if _HAS_APSCHEDULER:
            self._scheduler = _APScheduler()
            self._scheduler.add_job(
                self._run_classification,
                trigger="interval",
                seconds=classify_interval,
                id="vault_classification",
                misfire_grace_time=300,
                coalesce=True,
                max_instances=1,
            )
            self._scheduler.add_job(
                self._run_sync,
                trigger="interval",
                seconds=sync_interval,
                id="vault_sync",
                misfire_grace_time=300,
                coalesce=True,
                max_instances=1,
            )
            self._scheduler.start()
            logger.info(
                "APScheduler started: classify every %ds, sync every %ds",
                classify_interval, sync_interval,
            )
        else:
            # Fallback: raw asyncio loops (functional but no misfire recovery)
            asyncio.create_task(self._classification_loop())
            asyncio.create_task(self._sync_loop())
            logger.info(
                "Asyncio loops started: classify every %ds, sync every %ds",
                classify_interval, sync_interval,
            )

    async def _run_classification(self) -> None:
        """APScheduler job: run classification cycle."""
        try:
            await self.vault_engine.run_classification_cycle()
        except Exception as exc:
            logger.error("Classification cycle error: %s", exc)

    async def _run_sync(self) -> None:
        """APScheduler job: run vault sync."""
        try:
            import time
            await self.vault_sync.sync()
            self._last_sync_at = int(time.time())
        except Exception as exc:
            logger.error("Vault sync error: %s", exc)

    async def _classification_loop(self) -> None:
        """Asyncio fallback loop for classification."""
        while True:
            await asyncio.sleep(self.cfg.vault.classification_interval_seconds)
            await self._run_classification()

    async def _sync_loop(self) -> None:
        """Asyncio fallback loop for vault sync."""
        while True:
            await asyncio.sleep(self.cfg.vault.sync_interval_seconds)
            await self._run_sync()

    async def _knowledge_ingest(
        self, source: str, source_type: str = "auto", title: str | None = None
    ) -> dict:
        """Ingest an external document into the knowledge graph."""
        from openstinger.knowledge.ingest import ingest
        try:
            result = await ingest(
                source=source,
                agent_namespace=self.cfg.agent_namespace,
                driver=self.tier1.driver,
                embedder=self.tier1.embedder,
                source_type=source_type,
                title=title,
            )
            return {
                "document_uuid": result.document_uuid,
                "chunk_count": result.chunk_count,
                "source_type": result.source_type,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        except Exception as exc:
            logger.error("knowledge_ingest error: %s", exc)
            return {"error": str(exc)}

    async def _namespace_list(self, include_archived: bool = False) -> dict:
        """List registered agent namespaces."""
        from openstinger.agents.namespace import list_namespaces
        records = await list_namespaces(
            db=self.tier1.db, include_archived=include_archived
        )
        return {"namespaces": [r.to_dict() for r in records]}

    async def _namespace_create(self, name: str) -> dict:
        """Create a new agent namespace."""
        from openstinger.agents.namespace import create_namespace
        record = await create_namespace(
            name=name,
            db=self.tier1.db,
            driver=self.tier1.driver,
        )
        return record.to_dict()

    async def _namespace_archive(self, agent_id: str) -> dict:
        """Archive an agent namespace."""
        from openstinger.agents.namespace import archive_namespace
        success = await archive_namespace(agent_id=agent_id, db=self.tier1.db)
        return {"archived": success, "agent_id": agent_id}

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        await self.tier1.shutdown()


async def _run_stdio(cfg: Any) -> None:
    server = ScaffoldServer(cfg)
    await server.startup()
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.mcp.run(
                read_stream,
                write_stream,
                server.mcp.create_initialization_options(),
            )
    finally:
        await server.shutdown()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = load_config()
    asyncio.run(_run_stdio(cfg))


if __name__ == "__main__":
    main()
