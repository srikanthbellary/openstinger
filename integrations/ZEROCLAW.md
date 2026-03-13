# OpenStinger + ZeroClaw

ZeroClaw is an ultra-lean Rust agent runtime  3.4MB binary, sub-10ms boot, a swappable memory trait that explicitly supports external memory providers. OpenStinger is the canonical external memory backend for ZeroClaw's MCP trait.

## What OpenStinger adds to ZeroClaw

ZeroClaw lists `ephemeral` as a valid memory option  meaning sessions can be lost between restarts by design. OpenStinger replaces that with a persistent, queryable, graph-structured memory system that ZeroClaw never has to build.

| ZeroClaw alone | ZeroClaw + OpenStinger |
|---|---|
| Ephemeral or flat SQLite memory | Bi-temporal FalkorDB knowledge graph |
| No entity extraction | Named entities with deduplication |
| No knowledge distillation | Autonomous vault classification |
| 3.4MB binary stays tiny | Full memory stack offloaded to OpenStinger |

**The architecture story:** ZeroClaw handles lightweight local execution. OpenStinger runs on a server (even a \ VPS). ZeroClaw at near-zero RAM + OpenStinger = full enterprise memory at minimal edge cost.

## Setup

### 1. Start OpenStinger
Follow the [main README](../README.md). OpenStinger runs as an SSE server.

```bash
python -m openstinger.gradient.mcp.server
```

### 2. Configure ZeroClaw's MCP trait

In your ZeroClaw config (`config.toml` or equivalent):

```toml
[tools.mcp]
servers = [
  { name = "openstinger", url = "http://localhost:8766/sse" }
]
```

### 3. Configure session ingestion

```yaml
# OpenStinger config.yaml
ingestion:
  sessions_dir: "/path/to/zeroclaw/sessions"
  session_format: simple
```

## Why This Works for ZeroClaw

ZeroClaw's swappable memory trait was built for exactly this pattern. The binary stays 3.4MB. The memory system becomes a full knowledge graph on a separate process. No ZeroClaw source modifications. No RAM cost on the agent host.

For edge/IoT deployments: run ZeroClaw on \ hardware, point it at an OpenStinger instance on a central server. Multiple ZeroClaw devices share the same entity registry  they collectively build a shared knowledge graph.

## About Tier 3: Gradient

OpenStinger's Tier 3 (Gradient) evaluates agent alignment using a differential model:

> **dE/dt = β(C–D)E**

**E** = accumulated episodic memory · **β** = distilled self-knowledge · **C** = the agent's own defined constraints · **D** = observed deviation per response. Every output is scored against the agent's own evolving baseline. Starts in `observe_only = true`: always measuring, never blocking, until you configure otherwise.

---

## Integration Mode

→ **[Full integration modes guide with config snippets](INTEGRATION_MODES.md)**

**Recommended for ZeroClaw:** Skip Mode 1. Start with **Mode 2 (Primary)** or **Mode 3 (Exclusive)** immediately. ZeroClaw's ephemeral memory option means there's nothing worth preserving in the native layer — OpenStinger fills the gap from day one.

---

*For the full OpenStinger setup guide, see the [main README](../README.md).*
