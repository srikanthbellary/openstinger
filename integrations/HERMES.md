# OpenStinger + Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) is an open-source agent runtime with a native MCP client. OpenStinger attaches as an external MCP memory server: bi-temporal episodic memory, StingerVault notes, and Gradient alignment tools over the same SSE endpoint used by other frameworks.

Local evaluation clone (workspace): `harnesses/hermes-agent/` (upstream NousResearch/hermes-agent).

## What OpenStinger adds

| Without OpenStinger | With OpenStinger |
|---|---|
| Hermes built-in / skill memory | Bi-temporal episodic knowledge graph (FalkorDB) |
| Per-session context | Cross-session hybrid BM25 + vector recall |
| No structured self-model | StingerVault categories (identity, domain, method, prefs, constraints) |
| No operational audit trail | PostgreSQL event log |
| No alignment layer | Gradient evaluation (observe-only by default) |

## Setup

### 1. Start OpenStinger

Follow the [main README](../README.md) quick start. Default MCP SSE endpoint:

`http://localhost:8766/sse`

```bash
python -m openstinger.mcp.server
```

### 2. Register OpenStinger in Hermes MCP config

Hermes reads MCP servers from `~/.hermes/config.yaml` under `mcp_servers` (HTTP/SSE via `url`, or stdio via `command`/`args`).

**Recommended (SSE, matches other OpenStinger clients):**

```yaml
mcp_servers:
  openstinger:
    url: "http://localhost:8766/sse"
```

**Optional (stdio, if you prefer a subprocess launch):**

```yaml
mcp_servers:
  openstinger:
    command: "python"
    args: ["-m", "openstinger.mcp.server"]
    # Ensure OPENSTINGER / FalkorDB env vars are set in Hermes env or here:
    # env:
    #   ...
```

Reload MCP without restarting Hermes when supported:

```text
/reload-mcp
```

Or restart `hermes chat` / the gateway after editing config.

### 3. Confirm tools

Ask Hermes to list or use OpenStinger tools (for example `memory_wake_up`, `memory_query`, `memory_add`). Hermes discovers tools at MCP connect time.

Optional: constrain the tool surface with Hermes `tools.include` / `tools.exclude` if the full 32-tool list is too large for a given session.

### 4. Session ingestion (optional)

Point OpenStinger at Hermes session / workspace paths when you want automatic graph ingest of past chats. Hermes stores agent state under `~/.hermes` (see upstream docs). Configure OpenStinger:

```yaml
ingestion:
  sessions_dir: "/path/to/hermes/sessions"   # adjust to your layout
  session_format: simple                     # or openclaw if JSONL matches
  poll_interval_seconds: 10
```

Exact session file layout varies by Hermes version; verify under `~/.hermes` before enabling poll ingest.

## Integration mode

Same three modes as other frameworks: **Alongside** → **Primary** → **Exclusive**. See [INTEGRATION_MODES.md](INTEGRATION_MODES.md).

For Hermes, start **Alongside**: Hermes keeps its learning loop and skills; OpenStinger provides durable, queryable project memory over MCP.

## References

- Hermes MCP docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- OpenStinger tools and LongMemEval: [README](../README.md)
