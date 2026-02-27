"""
Tier 3 (Gradient) MCP server.

Routes all 15 Tier 1+2 tools unchanged + adds 5 Gradient tools + 3 ops tools = 23 total.

Gradient tools:
  gradient_status         — gradient health, profile state, observe_only flag
  gradient_alignment_score — evaluate a text and return score + verdict
  gradient_drift_status   — rolling window stats
  gradient_alignment_log  — recent evaluation log
  gradient_alert          — current alert status

Observability tools (v0.6):
  ops_status              — single-call dashboard: vault + classification + drift + alignment
  gradient_history        — last N alignment verdicts from PostgreSQL
  drift_status            — drift window history from PostgreSQL
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
    # --- v0.6 Observability Tools ---
    types.Tool(
        name="ops_status",
        description=(
            "Single-call operational dashboard. Returns vault note counts by category, "
            "last classification cycle stats, gradient alignment pass rate (last 20), "
            "and current drift state. Use this at session start for a full health check."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_namespace": {"type": "string", "default": "main"},
            },
        },
    ),
    types.Tool(
        name="gradient_history",
        description="Get last N alignment evaluation verdicts with scores from the operational DB.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "agent_namespace": {"type": "string", "default": "main"},
            },
        },
    ),
    types.Tool(
        name="drift_status",
        description="Get behavioral drift window history from the operational DB.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 5},
                "agent_namespace": {"type": "string", "default": "main"},
            },
        },
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
        # Route all Tier 1+2 tools to the Scaffold server
        if (name.startswith("memory_")
                or name.startswith("vault_")
                or name.startswith("knowledge_")
                or name.startswith("namespace_")):
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
                if not self.drift_detector:
                    return {}
                status = self.drift_detector.get_status()
                return {
                    "window_size": status.window_size,
                    "mean_score": status.mean_score,
                    "consecutive_flags": status.consecutive_flags,
                    "alert_active": status.alert_active,
                    "total_evaluated": status.total_evaluated,
                    "total_flagged": status.total_flagged,
                    "soft_flag_rate": round(status.soft_flag_rate, 4),
                }
            case "gradient_alignment_log":
                return await self._get_alignment_log(args.get("limit", 20))
            case "gradient_alert":
                return await self._get_alert_status()
            case "ops_status":
                return await self._ops_status(args.get("agent_namespace", self.cfg.agent_namespace))
            case "gradient_history":
                return await self._gradient_history(
                    args.get("agent_namespace", self.cfg.agent_namespace),
                    int(args.get("limit", 20)),
                )
            case "drift_status":
                return await self._drift_status(
                    args.get("agent_namespace", self.cfg.agent_namespace),
                    int(args.get("limit", 5)),
                )
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
        """Read recent evaluation events from alignment_events table."""
        try:
            rows = await self.tier2.tier1.db.get_alignment_events(
                agent_namespace=self.cfg.agent_namespace,
                limit=limit,
            )
            return [
                {
                    "event_uuid": r.uuid,
                    "verdict": r.verdict,
                    "scores": json.loads(r.scores_json) if r.scores_json else {},
                    "issues": json.loads(r.issues_json) if r.issues_json else [],
                    "corrected": bool(r.corrected),
                    "profile_state": r.profile_state,
                    "latency_ms": r.latency_ms,
                    "evaluated_at": r.evaluated_at,
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("gradient_alignment_log error: %s", exc)
            return []

    async def _get_alert_status(self) -> dict:
        """Read drift alert status from DriftDetector in-memory state."""
        if not self.drift_detector:
            return {"alert_active": False, "window_mean": 1.0, "consecutive_flags": 0,
                    "soft_flag_rate": 0.0, "total_evaluated": 0, "total_flagged": 0}
        status = self.drift_detector.get_status()
        return {
            "alert_active": status.alert_active,
            "window_mean": round(status.mean_score, 4),
            "consecutive_flags": status.consecutive_flags,
            "soft_flag_rate": round(status.soft_flag_rate, 4),
            "total_evaluated": status.total_evaluated,
            "total_flagged": status.total_flagged,
        }

    async def _ops_status(self, namespace: str) -> dict:
        """v0.6: Single-call operational dashboard."""
        db = self.tier2.tier1.db
        try:
            # Vault notes by category
            notes = await db.list_vault_notes(namespace)
            note_counts: dict = {}
            for n in notes:
                note_counts[n.category] = note_counts.get(n.category, 0) + 1

            # Last classification cycle
            class_log = await db.get_classification_history(namespace, limit=1)
            last_cycle = class_log[0] if class_log else None

            # Alignment pass rate (last 20)
            events = await db.get_alignment_events(namespace, limit=20)
            pass_rate = (
                sum(1 for e in events if e.verdict == "pass") / max(len(events), 1)
            )

            # Drift state (last entry)
            drift_rows = await db.get_drift_history(namespace, limit=1)
            last_drift = drift_rows[0] if drift_rows else None

            return {
                "vault_notes": note_counts,
                "total_active_notes": len(notes),
                "last_classification": {
                    "notes_created": last_cycle.notes_created if last_cycle else 0,
                    "notes_evolved": last_cycle.notes_evolved if last_cycle else 0,
                    "episodes_processed": last_cycle.episodes_processed if last_cycle else 0,
                    "completed_at": last_cycle.completed_at if last_cycle else None,
                } if last_cycle else None,
                "gradient": {
                    "alignment_pass_rate_last_20": round(pass_rate, 3),
                    "total_evaluated": len(events),
                    "drift_mean_score": round(last_drift.mean_score, 4) if last_drift else None,
                    "consecutive_flags": last_drift.consecutive_flags if last_drift else 0,
                    "alert_triggered": bool(last_drift.alert_triggered) if last_drift else False,
                },
                "gradient_observe_only": self.cfg.gradient.observe_only,
                "namespace": namespace,
            }
        except Exception as exc:
            logger.warning("ops_status error: %s", exc)
            return {"error": str(exc), "namespace": namespace}

    async def _gradient_history(self, namespace: str, limit: int = 20) -> list:
        """v0.6: Recent alignment verdicts from PostgreSQL."""
        db = self.tier2.tier1.db
        try:
            rows = await db.get_alignment_events(namespace, limit=limit)
            pass_count = sum(1 for r in rows if r.verdict == "pass")
            return {
                "events": [
                    {
                        "verdict": r.verdict,
                        "value_coherence_score": json.loads(r.scores_json or "{}").get(
                            "value_coherence", None
                        ),
                        "issues": json.loads(r.issues_json or "[]"),
                        "corrected": bool(r.corrected),
                        "evaluated_at": r.evaluated_at,
                        "latency_ms": r.latency_ms,
                    }
                    for r in rows
                ],
                "total": len(rows),
                "pass_rate": round(pass_count / max(len(rows), 1), 3),
            }
        except Exception as exc:
            logger.warning("gradient_history error: %s", exc)
            return {"error": str(exc)}

    async def _drift_status(self, namespace: str, limit: int = 5) -> dict:
        """v0.6: Drift window history from PostgreSQL."""
        db = self.tier2.tier1.db
        try:
            rows = await db.get_drift_history(namespace, limit=limit)
            return {
                "history": [
                    {
                        "mean_score": round(d.mean_score, 4),
                        "consecutive_flags": d.consecutive_flags,
                        "soft_flag_rate": round(d.soft_flag_rate, 4),
                        "total_evaluated": d.total_evaluated,
                        "alert_triggered": bool(d.alert_triggered),
                        "recorded_at": d.recorded_at,
                    }
                    for d in rows
                ],
                "current_in_memory": await self._get_alert_status(),
            }
        except Exception as exc:
            logger.warning("drift_status error: %s", exc)
            return {"error": str(exc)}

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
