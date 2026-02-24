"""
Tier 3 (Gradient) MCP server.

Routes all 15 Tier 1+2 tools unchanged + adds 5 Gradient tools = 20 total.

New tools:
  gradient_status         — gradient health, profile state, observe_only flag
  gradient_alignment_score — evaluate a text and return score + verdict
  gradient_drift_status   — rolling window stats
  gradient_alignment_log  — recent evaluation log
  gradient_alert          — current alert status
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from openstinger.config import load_config
from openstinger.gradient.alignment_profile import AlignmentProfileBuilder
from openstinger.gradient.correction_engine import CorrectionEngine
from openstinger.gradient.drift_detector import DriftDetector
from openstinger.gradient.interceptor import GradientInterceptor
from openstinger.scaffold.mcp.server import ScaffoldServer, ALL_TOOLS as TIER2_TOOLS

logger = logging.getLogger(__name__)

GRADIENT_TOOLS = [
    types.Tool(
        name="gradient_status",
        description="Get Gradient harness health, profile state, and observe_only flag.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="gradient_alignment_score",
        description="Evaluate a text response and return alignment score and verdict.",
        inputSchema={
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
            },
            "required": ["response_text"],
        },
    ),
    types.Tool(
        name="gradient_drift_status",
        description="Get rolling window alignment statistics and alert status.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="gradient_alignment_log",
        description="Get recent alignment evaluation log entries.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    types.Tool(
        name="gradient_alert",
        description="Get current drift alert status.",
        inputSchema={"type": "object", "properties": {}},
    ),
]

ALL_TOOLS = TIER2_TOOLS + GRADIENT_TOOLS


class GradientServer:
    """Tier 3 MCP server wrapping Tier 1+2 + adding Gradient tools."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mcp = Server("openstinger-gradient")
        self.tier2: Any = None
        self.interceptor: Any = None
        self.drift_detector: Any = None
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
        if name.startswith("memory_") or name.startswith("vault_"):
            return await self.tier2._dispatch(name, args)

        match name:
            case "gradient_status":
                return await self._gradient_status()
            case "gradient_alignment_score":
                result = await self.interceptor.evaluate(args.get("response_text", ""))
                return {
                    "verdict": result.verdict,
                    "scores": result.scores,
                    "issues": result.issues,
                    "latency_ms": result.latency_ms,
                    "corrected": result.corrected,
                }
            case "gradient_drift_status":
                return self.drift_detector.get_status().__dict__ if self.drift_detector else {}
            case "gradient_alignment_log":
                return await self._get_alignment_log(args.get("limit", 20))
            case "gradient_alert":
                return await self._get_alert_status()
            case _:
                return {"error": f"Unknown tool: {name}"}

    async def _gradient_status(self) -> dict:
        profile = self.interceptor._profile
        return {
            "enabled": self.cfg.gradient.enabled,
            "observe_only": self.cfg.gradient.observe_only,
            "profile_state": profile.state if profile else "not_built",
            "namespace": self.cfg.agent_namespace,
            "evaluation_timeout_ms": self.cfg.gradient.evaluation_timeout_ms,
        }

    async def _get_alignment_log(self, limit: int = 20) -> list:
        """Read recent evaluation events from session_state."""
        try:
            state = await self.tier2.tier1.db.get_session_state(self.cfg.agent_namespace)
            cursors = state.get_cursors()
            return cursors.get("gradient_events", [])[-limit:]
        except Exception:
            return []

    async def _get_alert_status(self) -> dict:
        try:
            state = await self.tier2.tier1.db.get_session_state(self.cfg.agent_namespace)
            cursors = state.get_cursors()
            alert = cursors.get("drift_alert")
            return {
                "alert_active": self.drift_detector._alert_active if self.drift_detector else False,
                "last_alert": alert,
            }
        except Exception:
            return {"alert_active": False}

    async def startup(self) -> None:
        cfg = self.cfg

        # Start Tier 2
        self.tier2 = ScaffoldServer(cfg)
        await self.tier2.startup()

        db = self.tier2.tier1.db
        driver = self.tier2.tier1.driver
        llm = self.tier2.tier1.llm

        # Drift detector
        self.drift_detector = DriftDetector(
            db=db,
            agent_namespace=cfg.agent_namespace,
            window_size=cfg.gradient.drift_window_size,
            alert_threshold=cfg.gradient.drift_alert_threshold,
            consecutive_flag_limit=cfg.gradient.consecutive_flag_limit,
        )

        # Interceptor
        self.interceptor = GradientInterceptor(
            llm=llm,
            driver=driver,
            db=db,
            agent_namespace=cfg.agent_namespace,
            observe_only=cfg.gradient.observe_only,
            evaluation_timeout_ms=cfg.gradient.evaluation_timeout_ms,
            drift_detector=self.drift_detector,
        )

        # Correction engine
        correction_engine = CorrectionEngine(llm=llm, interceptor=self.interceptor)
        self.interceptor.correction_engine = correction_engine

        # Build initial profile
        await self.interceptor.refresh_profile()

        # Hook profile refresh to vault sync
        original_sync = self.tier2._vault_sync_now
        async def sync_and_refresh() -> dict:
            result = await original_sync()
            await self.interceptor.refresh_profile()
            return result
        self.tier2._vault_sync_now = sync_and_refresh

        logger.info("Gradient server ready: namespace=%s observe_only=%s",
                    cfg.agent_namespace, cfg.gradient.observe_only)

    async def shutdown(self) -> None:
        await self.tier2.shutdown()


async def _run_stdio(cfg: Any) -> None:
    server = GradientServer(cfg)
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


async def _run_sse(cfg: Any) -> None:
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route
    from typing import Any as AnyType

    server = GradientServer(cfg)
    await server.startup()

    sse = SseServerTransport("/messages/")
    init_opts = server.mcp.create_initialization_options()

    async def handle_sse(request: AnyType) -> Response:
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
        uvi_config = uvicorn.Config(
            app, host="0.0.0.0", port=cfg.mcp.tcp_port, log_level="info"
        )
        uvi_server = uvicorn.Server(uvi_config)
        await uvi_server.serve()
    finally:
        await server.shutdown()


def main() -> None:
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
