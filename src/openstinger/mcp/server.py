"""
Tier 1 MCP server — OpenStinger memory harness.

Exposes 11 MCP tools. Supports stdio (default) and TCP transport.

Startup sequence:
  1. Load config (config.yaml + .env)
  2. Connect FalkorDB + init schema
  3. Init operational DB
  4. Build temporal engine + deduplicator + conflict resolver
  5. Warm up EntityRegistry (LSH index rebuild)
  6. Register agent namespace + start SessionReader
  7. Serve MCP
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from openstinger.config import HarnessConfig, load_config
from openstinger.ingestion.scheduler import IngestionSchedulerRegistry
from openstinger.mcp.tools.memory_tools import (
    memory_add,
    memory_delete,
    memory_get_entity,
    memory_get_episode,
    memory_ingest_now,
    memory_job_status,
    memory_list_agents,
    memory_namespace_status,
    memory_query,
    memory_search,
    memory_update,
)
from openstinger.operational.adapter import create_adapter
from openstinger.temporal.anthropic_client import AnthropicClient
from openstinger.temporal.openai_compatible_client import OpenAICompatibleClient
from openstinger.temporal.conflict_resolver import ConflictResolver
from openstinger.temporal.deduplicator import DeduplicationEngine
from openstinger.temporal.engine import TemporalEngine
from openstinger.temporal.entity_registry import EntityRegistry
from openstinger.temporal.falkordb_driver import wait_for_falkordb
from openstinger.temporal.openai_embedder import OpenAIEmbedder
from openstinger.storage.embedding_cache import CachedEmbedder, EmbeddingCache

logger = logging.getLogger(__name__)

# MCP tool schemas
TOOL_SCHEMAS: list[types.Tool] = [
    types.Tool(
        name="memory_add",
        description="Add an episode to the agent's temporal memory graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Episode text content"},
                "source": {"type": "string", "default": "manual"},
                "source_description": {"type": "string", "default": ""},
                "valid_at_unix": {"type": "integer", "description": "Unix timestamp when episode occurred"},
                "agent_namespace": {"type": "string"},
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_query",
        description=(
            "Hybrid semantic search (BM25 + vector) across episodes, entities, and facts. "
            "Returns a unified 'ranked' list with normalized scores (all comparable 0.0–1.0). "
            "Use after_date / before_date to restrict to a time window (e.g. after_date='2026-02')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "include_expired": {"type": "boolean", "default": False},
                "agent_namespace": {"type": "string"},
                "after_date": {
                    "type": "string",
                    "description": "ISO date filter — only return episodes on/after this date. Format: YYYY-MM-DD or YYYY-MM",
                },
                "before_date": {
                    "type": "string",
                    "description": "ISO date filter — only return episodes on/before this date. Format: YYYY-MM-DD or YYYY-MM",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_search",
        description=(
            "Smart keyword search across episodes, entities, or facts. "
            "Automatically handles: numeric data (IP addresses, prices, wallet addresses), "
            "temporal queries (month/year searches like 'February 2026'), "
            "fuzzy matching (typos — 'Qinn' finds 'Quinn' via vector fallback). "
            "Falls back from BM25 → vector → CONTAINS if primary search returns no results. "
            "Supports date filtering via after_date / before_date."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "search_type": {
                    "type": "string",
                    "enum": ["episodes", "entities", "facts", "all"],
                    "default": "all",
                },
                "limit": {"type": "integer", "default": 10},
                "agent_namespace": {"type": "string"},
                "after_date": {
                    "type": "string",
                    "description": "ISO date — only return results on/after this date. Format: YYYY-MM-DD or YYYY-MM",
                },
                "before_date": {
                    "type": "string",
                    "description": "ISO date — only return results on/before this date. Format: YYYY-MM-DD or YYYY-MM",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_get_entity",
        description="Fetch an entity node and its current edges by UUID.",
        inputSchema={
            "type": "object",
            "properties": {"uuid": {"type": "string"}},
            "required": ["uuid"],
        },
    ),
    types.Tool(
        name="memory_get_episode",
        description="Fetch an episode node by UUID.",
        inputSchema={
            "type": "object",
            "properties": {"uuid": {"type": "string"}},
            "required": ["uuid"],
        },
    ),
    types.Tool(
        name="memory_job_status",
        description="Check the status of an ingestion job.",
        inputSchema={
            "type": "object",
            "properties": {"job_uuid": {"type": "string"}},
            "required": ["job_uuid"],
        },
    ),
    types.Tool(
        name="memory_ingest_now",
        description="Trigger immediate ingestion from session files (bypasses poll interval).",
        inputSchema={
            "type": "object",
            "properties": {"agent_namespace": {"type": "string"}},
        },
    ),
    types.Tool(
        name="memory_namespace_status",
        description="Get health and statistics for an agent namespace.",
        inputSchema={
            "type": "object",
            "properties": {"agent_namespace": {"type": "string"}},
        },
    ),
    types.Tool(
        name="memory_list_agents",
        description="List all registered agent namespaces.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="memory_delete",
        description=(
            "Permanently delete an episode from the temporal memory graph. "
            "Cascade-removes any entity nodes that become orphaned (no remaining episode connections). "
            "Use to remove stale, incorrect, or sensitive memories."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "episode_uuid": {"type": "string", "description": "UUID of the episode to delete"},
            },
            "required": ["episode_uuid"],
        },
    ),
    types.Tool(
        name="memory_update",
        description=(
            "Update the content of an existing episode and re-index it. "
            "Re-embeds the new content and runs entity extraction diff (adds new entities, "
            "keeps existing). Use to correct or enrich a stored memory."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "episode_uuid": {"type": "string", "description": "UUID of the episode to update"},
                "new_content": {"type": "string", "description": "Replacement content for the episode"},
            },
            "required": ["episode_uuid", "new_content"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

class OpenStingerServer:
    """Assembled Tier 1 MCP server."""

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.mcp = Server("openstinger")

        # Components (set during startup)
        self.driver: Any = None
        self.db: Any = None
        self.llm: Any = None
        self.embedder: Any = None
        self.entity_registry: Any = None
        self.engine: Any = None
        self.scheduler: Any = None

        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.mcp.list_tools()
        async def list_tools() -> list[types.Tool]:
            return TOOL_SCHEMAS

        @self.mcp.call_tool()
        async def call_tool(
            name: str, arguments: dict
        ) -> list[types.TextContent]:
            result = await self._dispatch(name, arguments)
            import json
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _dispatch(self, name: str, args: dict) -> Any:
        match name:
            case "memory_add":
                return await memory_add(self.engine, self.db, **args)
            case "memory_query":
                return await memory_query(self.engine, **args)
            case "memory_search":
                return await memory_search(self.engine, **args)
            case "memory_get_entity":
                return await memory_get_entity(self.engine, **args)
            case "memory_get_episode":
                return await memory_get_episode(self.engine, **args)
            case "memory_job_status":
                return await memory_job_status(self.db, **args)
            case "memory_ingest_now":
                return await memory_ingest_now(self.scheduler, **args)
            case "memory_namespace_status":
                return await memory_namespace_status(self.engine, self.db, **args)
            case "memory_list_agents":
                return await memory_list_agents(self.scheduler)
            case "memory_delete":
                return await memory_delete(self.engine, self.db, **args)
            case "memory_update":
                return await memory_update(self.engine, **args)
            case _:
                return {"error": f"Unknown tool: {name}"}

    async def startup(self) -> None:
        """Full startup sequence."""
        cfg = self.cfg
        logger.info("OpenStinger starting up: agent=%s", cfg.agent_name)

        # 1. FalkorDB
        self.driver = await wait_for_falkordb(
            host=cfg.falkordb.host,
            port=cfg.falkordb.port,
            password=cfg.falkordb.password,
            timeout_seconds=30.0,
            vector_dimensions=cfg.falkordb.vector_dimensions,
        )
        await self.driver.init_schema()

        # 2. Operational DB
        self.db = create_adapter(
            provider=cfg.operational_db.provider,
            sqlite_path=cfg.resolved_sqlite_path(),
            postgresql_url=cfg.operational_db.postgresql_url,
        )
        await self.db.init()

        # 3. LLM + Embedder
        if cfg.llm.llm_base_url:
            # OpenAI-compatible provider (Novita, DeepSeek, etc.)
            self.llm = OpenAICompatibleClient(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=cfg.llm.model,
                fast_model=cfg.llm.fast_model,
                base_url=cfg.llm.llm_base_url,
            )
        else:
            self.llm = AnthropicClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model=cfg.llm.model,
                fast_model=cfg.llm.fast_model,
            )
        if cfg.llm.embedding_provider == "ollama":
            self.embedder = OpenAIEmbedder(
                api_key="ollama",
                model=cfg.llm.embedding_model,
                dimensions=cfg.falkordb.vector_dimensions,
                base_url=f"{cfg.llm.ollama_host}/v1",
                skip_dimensions=True,
            )
            logger.info("Embedder: Ollama  host=%s  model=%s  dims=%d",
                        cfg.llm.ollama_host, cfg.llm.embedding_model, cfg.falkordb.vector_dimensions)
        else:
            self.embedder = OpenAIEmbedder(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=cfg.llm.embedding_model,
                dimensions=cfg.falkordb.vector_dimensions,
                base_url=cfg.llm.embedding_base_url or None,
            )
            logger.info("Embedder: %s  model=%s  dims=%d",
                        cfg.llm.embedding_provider, cfg.llm.embedding_model, cfg.falkordb.vector_dimensions)

        # v0.5: wrap embedder with SQLite-backed cache to eliminate redundant
        # API calls during vault re-syncs and repeated entity embedding.
        _cache_db = cfg.resolved_sqlite_path().parent / "embed_cache.db"
        _embed_cache = EmbeddingCache(db_path=_cache_db, model_name=cfg.llm.embedding_model)
        await _embed_cache.init()
        self.embedder = CachedEmbedder(embedder=self.embedder, cache=_embed_cache)
        logger.info("EmbeddingCache initialised: %s", _cache_db)

        # 4. Entity registry + temporal engine
        self.entity_registry = EntityRegistry(self.db)
        await self.entity_registry.warmup()

        self.engine = TemporalEngine(
            driver=self.driver,
            llm=self.llm,
            embedder=self.embedder,
            entity_registry=self.entity_registry,
            agent_namespace=cfg.agent_namespace,
        )

        deduplicator = DeduplicationEngine(
            llm=self.llm,
            lsh_threshold=cfg.deduplication.lsh_threshold,
            llm_confidence_min=cfg.deduplication.llm_confidence_min,
            token_overlap_min=cfg.deduplication.token_overlap_min,
        )
        await deduplicator.rebuild_lsh_index(self.driver, cfg.agent_namespace)
        self.engine.set_deduplicator(deduplicator)

        conflict_resolver = ConflictResolver(llm=self.llm, driver=self.driver)
        self.engine.set_conflict_resolver(conflict_resolver)

        # 5. Ingestion scheduler
        self.scheduler = IngestionSchedulerRegistry()
        await self.scheduler.register_agent(
            namespace=cfg.agent_namespace,
            sessions_dir=cfg.resolved_sessions_dir(),
            profile_dirs=cfg.resolved_profile_dirs(),
            engine=self.engine,
            db_adapter=self.db,
            poll_interval=cfg.ingestion.poll_interval_seconds,
            chunk_size=cfg.ingestion.chunk_size,
            session_format=cfg.ingestion.session_format,
            concurrency=cfg.ingestion.concurrency,
        )

        logger.info("OpenStinger ready: namespace=%s", cfg.agent_namespace)

    async def shutdown(self) -> None:
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.driver:
            await self.driver.close()
        if self.db:
            await self.db.close()
        logger.info("OpenStinger shut down")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def _run_stdio(cfg: HarnessConfig) -> None:
    server = OpenStingerServer(cfg)
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


async def _run_sse(cfg: HarnessConfig) -> None:
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    server = OpenStingerServer(cfg)
    await server.startup()

    sse = SseServerTransport("/messages/")
    init_opts = server.mcp.create_initialization_options()

    async def handle_sse(request: Any) -> Response:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.mcp.run(streams[0], streams[1], init_opts)
        return Response()

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ])

    try:
        config = uvicorn.Config(
            app, host="0.0.0.0", port=cfg.mcp.tcp_port, log_level="info"
        )
        uvi_server = uvicorn.Server(config)
        await uvi_server.serve()
    finally:
        await server.shutdown()


def main() -> None:
    """CLI entry point: openstinger"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    cfg = load_config()

    if cfg.mcp.transport == "sse":
        asyncio.run(_run_sse(cfg))
    else:
        asyncio.run(_run_stdio(cfg))


if __name__ == "__main__":
    main()
