# OpenStinger + PicoClaw

PicoClaw is the lightweight, resource-constrained member of the \*Claw ecosystem — designed for edge deployment, low-memory environments, and single-purpose agent runtimes. OpenStinger connects to PicoClaw via MCP/SSE exactly as it does with any other \*Claw variant.

## What OpenStinger adds to PicoClaw

PicoClaw deliberately ships without a persistent memory layer to stay lightweight. OpenStinger fills that gap without adding runtime overhead to the PicoClaw process itself — all memory, knowledge, and alignment work runs in a separate Python process.

| Capability | PicoClaw (default) | PicoClaw + OpenStinger |
|---|---|---|
| Session memory | None | Full temporal graph (episodes + entities + facts) |
| Knowledge distillation | None | Autonomous StingerVault classification |
| Entity deduplication | None | 3-stage: string → MinHash LSH → LLM |
| Alignment evaluation | None | Gradient behavioral scoring |
| Memory portability | PicoClaw-only | Any MCP runtime (Tier 1–3) |

## Setup

### 1. Start OpenStinger

```bash
docker compose up -d   # starts FalkorDB, PostgreSQL, Adminer
python -m openstinger.gradient.mcp.server   # Tier 3 (all 30 tools)
```

For Tier 1 only (memory, no vault/gradient):
```bash
python -m openstinger.mcp.server
```

### 2. Configure MCP connection in PicoClaw

In PicoClaw's MCP config, add OpenStinger as an SSE connection:

```json
{
  "connections": {
    "openstinger": {
      "type": "sse",
      "url": "http://localhost:8766/sse"
    }
  }
}
```

> If PicoClaw runs inside Docker, use `host.docker.internal` instead of `localhost`.

### 3. Configure session ingestion

If PicoClaw writes session files, point OpenStinger at the sessions directory:

```yaml
# config.yaml
ingestion:
  sessions_dir: "/path/to/picoclaw/sessions"
  session_format: simple   # PicoClaw uses simple line-by-line JSONL
```

For lightweight deployments without a sessions directory, use `memory_add` directly from PicoClaw tools to store episodes manually.

### 4. Add the OpenStinger skill to your PicoClaw agent

Copy the canonical skill template and adapt it for your agent:

```bash
cp /path/to/openstinger/integrations/AGENT_SKILL_TEMPLATE.md \
   /path/to/picoclaw/workspace/skills/openstinger_skill.md
```

See [AGENT_SKILL_TEMPLATE.md](AGENT_SKILL_TEMPLATE.md) for the full tool reference and usage patterns.

## Recommended Tier for PicoClaw

Since PicoClaw targets minimal footprint, **Tier 1** (9 memory tools) is the recommended starting point. Upgrade to Tier 2 or 3 only when vault classification or alignment scoring is needed.

| Tier | Server | Tools | Use case |
|---|---|---|---|
| Tier 1 | `openstinger.mcp.server` | 11 | Memory only — minimal overhead |
| Tier 2 | `openstinger.scaffold.mcp.server` | 22 | Memory + StingerVault knowledge distillation |
| Tier 3 | `openstinger.gradient.mcp.server` | 30 | Full: memory + vault + behavioral alignment |

## Verifying the Connection

From inside a PicoClaw session, call:

```
openstinger.ops_status()
```

A healthy response looks like:

```json
{
  "vault": { "active_notes": 12, "last_sync": "2026-03-01T14:22:00Z" },
  "gradient": { "profile_state": "minimal", "observe_only": true },
  "drift": { "alert_active": false, "window_mean": 0.87 }
}
```

## Notes

- PicoClaw's low-memory design means the OpenStinger MCP server should run as a separate host process (not in the same container/process).
- The `simple` session format (`session_format: simple`) expects one JSON object per line with at minimum a `content` field and optionally a `timestamp` field.
- All 30 MCP tools are available regardless of PicoClaw's tier — the tool tier refers to OpenStinger's server startup, not PicoClaw's configuration.
