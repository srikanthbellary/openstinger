"""
Tier 3 (Gradient) MCP server.

Routes all 22 Tier 1+2 tools unchanged + adds 8 Gradient tools + 1-2 Honeypot tools = 31-32 total.

Gradient tools:
  gradient_status          — gradient health, profile state, observe_only flag
  gradient_alignment_score — evaluate a text and return score + verdict
  gradient_drift_status    — rolling window stats
  gradient_alignment_log   — recent evaluation log
  gradient_alert           — current alert status

Observability tools (v0.6):
  ops_status              — single-call dashboard: vault + classification + drift + alignment
  gradient_history        — last N alignment verdicts from PostgreSQL
  drift_status            — drift window history from PostgreSQL

Honeypot tools (v0.9):
  gradient_honeypot_status — registered patterns, recent alerts, lockdown state
  gradient_honeypot_clear  — (only registered when lockdown_mode: true) clears lockdown
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
from openstinger.gradient.honeypot.detector import HoneypotDetector
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
    # v0.9 Honeypot tools
    types.Tool(
        name="gradient_honeypot_status",
        description=(
            "Returns the current honeypot state: registered pattern counts, recent alerts, "
            "and whether lockdown is active. Use this after any anomalous retrieval behaviour "
            "to check whether a probe was detected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "agent_namespace": {"type": "string", "default": "main"},
                "limit": {"type": "integer", "default": 10, "description": "Recent alerts to return"},
            },
        },
    ),
]

ALL_TOOLS = TIER2_TOOLS + GRADIENT_TOOLS


class GradientServer:
    """Tier 3 MCP server wrapping Tier 1+2 + adding Gradient + Honeypot tools."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mcp = Server("openstinger-gradient")
        self.tier2: Any = None
        self.interceptor: Any = None
        self.drift_detector: Any = None
        self.honeypot_detector: Any = None  # v0.9
        self._lockdown_active: bool = False  # v0.9
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.mcp.list_tools()
        async def list_tools() -> list[types.Tool]:
            return ALL_TOOLS

        @self.mcp.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            # v0.9: lockdown check — refuse all calls when lockdown is active
            if self._lockdown_active and name not in ("gradient_honeypot_clear", "gradient_honeypot_status"):
                result = {"error": "honeypot_lockdown", "message": "System locked down due to honeypot trigger. Use gradient_honeypot_clear to resume."}
                return [types.TextContent(type="text", text=json.dumps(result))]
            result = await self._dispatch(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _dispatch(self, name: str, args: dict) -> Any:
        # v0.9: honeypot pre-execution hook on retrieval tools
        if name in ("memory_search", "memory_get_entity", "knowledge_search"):
            if self.honeypot_detector is not None:
                ns = args.get("agent_namespace") or self.cfg.agent_namespace
                query = args.get("query") or args.get("uuid") or ""
                alert = self.honeypot_detector.check(query, tool=name, agent_namespace=ns)
                if alert:
                    await self._handle_honeypot_alert(alert)
                    if self.cfg.gradient.honeypot.suppress_response:
                        return {"results": [], "honeypot_triggered": True}

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
            case "gradient_honeypot_status":
                return await self._honeypot_status(
                    args.get("agent_namespace", self.cfg.agent_namespace),
                    int(args.get("limit", 10)),
                )
            case "gradient_honeypot_clear":
                return await self._honeypot_clear(args.get("auth_code", ""))
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
            # v0.9: honeypot state (spec §7 — no breaking change, new fields only)
            "honeypot_enabled":         self.cfg.gradient.honeypot.enabled,
            "honeypot_lockdown_active": self._lockdown_active,
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
            # Vault notes by category — primary source: PostgreSQL
            notes = await db.list_vault_notes(namespace)
            note_counts: dict = {}
            for n in notes:
                note_counts[n.category] = note_counts.get(n.category, 0) + 1

            # PostgreSQL vault sync may lag behind FalkorDB (e.g. after direct vault_note_add).
            # If PostgreSQL shows 0 notes, fall back to FalkorDB category counts.
            if not note_counts:
                try:
                    vault_engine = self.tier2.vault_engine
                    if vault_engine:
                        stats = await vault_engine.get_vault_stats()
                        for cat, v in stats.items():
                            active = v.get("active", 0)
                            if active:
                                note_counts[cat] = active
                        if note_counts:
                            logger.info(
                                "ops_status: PostgreSQL vault empty for ns=%s, using FalkorDB counts",
                                namespace,
                            )
                except Exception as fb_exc:
                    logger.debug("ops_status FalkorDB fallback error: %s", fb_exc)

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

        # v0.9: Honeypot detector
        self.honeypot_detector = HoneypotDetector(cfg)

        # Register gradient_honeypot_clear only when lockdown_mode = true
        if cfg.gradient.honeypot.lockdown_mode:
            ALL_TOOLS.append(
                types.Tool(
                    name="gradient_honeypot_clear",
                    description=(
                        "Clear an active honeypot lockdown. Requires the operator auth_code "
                        "set in gradient.honeypot.lockdown_auth_code. Only available when "
                        "lockdown_mode: true."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "auth_code": {"type": "string", "description": "Operator auth code to clear lockdown"},
                        },
                        "required": ["auth_code"],
                    },
                )
            )

        logger.info("Gradient server ready: namespace=%s observe_only=%s honeypot=%s",
                    cfg.agent_namespace, cfg.gradient.observe_only,
                    "enabled" if cfg.gradient.honeypot.enabled else "disabled")

    async def shutdown(self) -> None:
        await self.tier2.shutdown()

    # ------------------------------------------------------------------
    # v0.9 Honeypot handlers
    # ------------------------------------------------------------------

    async def _handle_honeypot_alert(self, alert: Any) -> None:
        """
        Persist alert to DB, write hard_block alignment event, optionally set lockdown.
        Per spec §4.3 steps 1-4.
        """
        db = self.tier2.tier1.db
        ns = alert.agent_namespace
        suppressed = bool(self.cfg.gradient.honeypot.suppress_response)
        lockdown_triggered = False

        # 1. Write honeypot_alerts row
        try:
            await db.log_honeypot_alert(
                agent_namespace    = ns,
                tool_called        = alert.tool_called,
                query_text         = alert.query_text,
                matched_pattern    = alert.matched_pattern,
                pattern_source     = alert.pattern_source,
                suppressed         = suppressed,
                lockdown_triggered = lockdown_triggered,
            )
        except Exception as exc:
            logger.error("Failed to persist honeypot_alert: %s", exc)

        # 2. Write alignment_events row with verdict=hard_block / violation_type=honeypot
        try:
            await db.log_alignment_event(
                agent_namespace = ns,
                verdict         = "hard_block",
                scores          = {},
                issues          = [f"honeypot:{alert.matched_pattern}({alert.pattern_source})"],
                corrected       = False,
                profile_state   = "honeypot",
            )
        except Exception as exc:
            logger.error("Failed to write honeypot alignment_event: %s", exc)

        # 3. Feed into drift detector (counts as hard_block event)
        if self.drift_detector is not None:
            try:
                self.drift_detector.record(score=0.0, verdict="hard_block")
            except Exception:
                pass

        # 4. Lockdown if configured
        if self.cfg.gradient.honeypot.lockdown_mode:
            self._lockdown_active = True
            logger.error(
                "HONEYPOT LOCKDOWN ACTIVATED for namespace=%s — all tools blocked until cleared.", ns
            )

    async def _honeypot_status(self, agent_namespace: str, limit: int = 10) -> dict:
        """Handler for gradient_honeypot_status tool."""
        db = self.tier2.tier1.db
        detector = self.honeypot_detector

        # Recent alerts
        recent_alerts: list[dict] = []
        total_alerts = 0
        try:
            rows = await db.get_honeypot_alerts(agent_namespace, limit=limit)
            total_count_rows = await db.get_honeypot_alerts(agent_namespace, limit=10_000)
            total_alerts = len(total_count_rows)
            recent_alerts = [
                {
                    "tool_called":     r.tool_called,
                    "query_text":      r.query_text,
                    "matched_pattern": r.matched_pattern,
                    "pattern_source":  r.pattern_source,
                    "suppressed":      bool(r.suppressed),
                    "created_at":      r.created_at,
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("honeypot_status DB read error: %s", exc)

        return {
            "enabled":         detector.enabled if detector else False,
            "lockdown_active": self._lockdown_active,
            "pattern_count":   detector.pattern_count if detector else {"default": 0, "custom": 0},
            "recent_alerts":   recent_alerts,
            "total_alerts_all_time": total_alerts,
        }

    async def _honeypot_clear(self, auth_code: str) -> dict:
        """Handler for gradient_honeypot_clear tool. Only effective when lockdown_mode=true."""
        if not self.cfg.gradient.honeypot.lockdown_mode:
            return {"error": "lockdown_mode is not enabled in config — nothing to clear"}
        expected = self.cfg.gradient.honeypot.lockdown_auth_code or ""
        if not expected:
            return {"error": "lockdown_auth_code not set in gradient.honeypot config"}
        if auth_code != expected:
            return {"error": "invalid auth_code"}
        self._lockdown_active = False
        logger.info("Honeypot lockdown cleared by operator (auth_code accepted)")
        return {"success": True, "lockdown_active": False}




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
    import contextlib
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route
    from typing import Any as AnyType

    server = GradientServer(cfg)
    await server.startup()

    # --- Legacy SSE transport (Cursor, Claude Desktop, etc.) ---
    sse = SseServerTransport("/messages/")
    init_opts = server.mcp.create_initialization_options()

    async def handle_sse(request: AnyType) -> Response:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.mcp.run(streams[0], streams[1], init_opts)
        return Response()

    # --- Streamable HTTP transport (mcporter, Claude Code, newer clients) ---
    session_manager = StreamableHTTPSessionManager(
        app=server.mcp,
        stateless=True,  # each request is independent — no resumability needed
    )

    async def handle_streamable_http(request: AnyType) -> None:
        await session_manager.handle_request(request.scope, request.receive, request._send)

    @contextlib.asynccontextmanager
    async def lifespan(app: AnyType):
        async with session_manager.run():
            yield

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
            Route("/mcp", endpoint=handle_streamable_http, methods=["GET", "POST", "DELETE"]),
        ],
    )

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
