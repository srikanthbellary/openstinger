"""
Tier 2 (Scaffold) MCP server.

Routes all 9 Tier 1 tools unchanged + adds 6 new Scaffold tools = 15 total.

New tools:
  vault_status      — vault health and last sync time
  vault_sync_now    — trigger immediate vault file sync
  vault_stats       — note counts by category
  vault_promote_now — trigger immediate classification cycle
  vault_note_list   — list notes (filtered by category)
  vault_note_get    — fetch a single note by UUID
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
        name="vault_note_get",
        description="Fetch a single vault note by UUID.",
        inputSchema={
            "type": "object",
            "properties": {"uuid": {"type": "string"}},
            "required": ["uuid"],
        },
    ),
]

ALL_TOOLS = TIER1_TOOLS + TIER2_TOOLS


class ScaffoldServer:
    """Tier 2 MCP server wrapping Tier 1 + adding vault tools."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mcp = Server("openstinger-scaffold")
        self.tier1: Any = None        # OpenStingerServer
        self.vault_engine: Any = None
        self.vault_sync: Any = None
        self._last_sync_at: int = 0
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

        # Start background cron tasks
        asyncio.create_task(self._classification_loop())
        asyncio.create_task(self._sync_loop())

        logger.info("Scaffold server ready: namespace=%s", cfg.agent_namespace)

    async def _classification_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.vault.classification_interval_seconds)
            try:
                await self.vault_engine.run_classification_cycle()
            except Exception as exc:
                logger.error("Classification cycle error: %s", exc)

    async def _sync_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.vault.sync_interval_seconds)
            try:
                import time
                await self.vault_sync.sync()
                self._last_sync_at = int(time.time())
            except Exception as exc:
                logger.error("Vault sync error: %s", exc)

    async def shutdown(self) -> None:
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
