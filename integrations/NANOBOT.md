# OpenStinger + Nanobot

Nanobot is a research-friendly, 4,000-line Python agent runtime with MCP support since February 14, 2026. Its architecture is intentionally minimal  core stays small, tools plug in via MCP. OpenStinger is the memory backend it was designed to delegate to.

## What OpenStinger adds to Nanobot

| Without OpenStinger | With OpenStinger |
|---|---|
| Basic session memory | Bi-temporal episodic knowledge graph |
| No entity tracking | 3-stage entity extraction + deduplication |
| No knowledge distillation | Autonomous StingerVault classification |
| No audit trail | PostgreSQL operational event log |
| No alignment layer | Gradient alignment evaluation per response |

## Setup (3 steps)

### 1. Start OpenStinger
Follow the [main README](../README.md) quick start. OpenStinger runs as an SSE server on port 8766.

```bash
python -m openstinger.gradient.mcp.server
```

### 2. Add OpenStinger to Nanobot's MCP config

In your Nanobot MCP config (`mcpServers` section):

```json
{
  "mcpServers": {
    "openstinger": {
      "url": "http://localhost:8766/sse"
    }
  }
}
```

### 3. Configure session ingestion

In OpenStinger's `config.yaml`, point to Nanobot's session files:

```yaml
ingestion:
  sessions_dir: "/path/to/nanobot/sessions"
  session_format: simple   # or openclaw if Nanobot uses JSONL
  poll_interval_seconds: 10
```

## Available Tools

After connecting, your Nanobot agent has access to all 27 OpenStinger MCP tools:

- `memory_query` / `memory_search`  Hybrid BM25 + vector search across all sessions
- `vault_note_list` / `vault_note_get`  Structured self-knowledge by category
- `ops_status`  Full operational health in one call
- `gradient_history`  Alignment verdict log
- And 21 more  see the [full tools reference](../README.md#mcp-tools)

## Why This Works for Nanobot

Nanobot's philosophy is **small core, external everything**. OpenStinger is the external memory system that Nanobot always assumed would exist but never needed to build. Nanobot's codebase stays at 4,000 lines. OpenStinger handles the hard parts: entity graph, knowledge classification, audit logging, and alignment evaluation.

No Nanobot source code changes required. The codebase stays small. The memory system becomes enterprise-grade.

---

*For the full OpenStinger setup guide, see the [main README](../README.md).*
