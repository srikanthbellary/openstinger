# OpenStinger Integration Modes

> **Zero switching cost. Zero migration risk.**
> Add OpenStinger to whatever *Claw framework you're running today.
> Start alongside. Go primary when ready. No framework has to be abandoned.

This guide covers the three adoption modes for connecting OpenStinger to any MCP-compatible agent runtime — OpenClaw, Nanobot, ZeroClaw, NanoClaw, PicoClaw, or any *Claw framework that ships with built-in memory.

---

## The Three Modes

| Mode | Description | When to use |
|---|---|---|
| **Mode 1: Alongside** | OpenStinger runs next to the framework's native memory. Both active simultaneously. | Day 1 — zero disruption, additive only |
| **Mode 2: Primary** | Agent queries OpenStinger first. Falls back to native memory only if OpenStinger returns empty. | After OpenStinger has enough session history to be trusted |
| **Mode 3: Exclusive** | Disable the framework's native memory entirely. OpenStinger is the only memory backend. | Full commitment — cleanest, most coherent setup |

---

## Mode 1: Alongside (Recommended Start)

Run OpenStinger as an additional memory layer without touching your existing setup. The agent has access to both its framework's native memory tools AND all 27 OpenStinger MCP tools simultaneously.

### Setup

1. Start OpenStinger (see [main README](../README.md))
2. Add OpenStinger to your MCP config — same pattern for all frameworks:

```json
{
  "mcpServers": {
    "openstinger": {
      "url": "http://localhost:8766/sse"
    }
  }
}
```

3. Point session ingestion at your agent's session files:

```yaml
# config.yaml
ingestion:
  sessions_dir: "/path/to/your/agent/sessions"
  session_format: openclaw   # or: simple
```

### What happens

- Your agent's existing memory tools (`memory_search`, `memory_get`, etc.) keep working untouched
- OpenStinger's 28 tools are available alongside them
- OpenStinger ingests session files in the background and builds its knowledge graph
- You can call `memory_query` or `memory_search` from OpenStinger at any time
- Both systems accumulate history independently

### For ZeroClaw users
ZeroClaw's ephemeral memory option means sessions are lost between restarts. Running OpenStinger alongside ZeroClaw gives you the persistent memory layer that ZeroClaw explicitly outsources via its swappable trait.

---

## Mode 2: Primary (Recommended at Scale)

OpenStinger becomes the first search. The framework's native memory is the fallback. The agent always queries OpenStinger first — if results are found, they're used. If not, native memory is queried as a fallback.

This is purely a **system prompt / AGENTS.md instruction change** — no config changes required.

### Setup via system prompt

Add to your agent's `AGENTS.md` or system prompt:

```
When searching memory or recalling past conversations, decisions, or facts:
1. FIRST query OpenStinger: call memory_query or memory_search with your query
2. If OpenStinger returns results (episodes or entities found), use those results
3. If OpenStinger returns empty results, fall back to the framework's native memory_search tool
```

### Setup via MCP config (advanced — builds a chained skill)

For frameworks that support custom MCP tool wrapping (NanoClaw, ZeroClaw trait pattern):

```javascript
// unified-memory skill
async function search_memory(query) {
  // 1. Query OpenStinger first
  const openstingerResults = await mcpClient.call('openstinger.memory_query', {
    query: query,
    agent_namespace: 'main',
    limit: 5
  });

  if (openstingerResults.episodes?.length > 0) {
    return formatResults(openstingerResults);
  }

  // 2. Fall back to framework native memory
  return await tools.memory_search({ query });
}
```

### OpenClaw-specific: mcporter priority config

For OpenClaw agents using mcporter, add to your agent's system prompt or `AGENTS.md`:

```
Before using OpenClaw's memory_search tool, FIRST query OpenStinger:
- Run: mcporter call openstinger.memory_search with your query
- If OpenStinger returns results, use those
- If OpenStinger returns empty, fall back to memory_search
```

### Why make OpenStinger primary?

OpenStinger's search is graph-aware and entity-conditioned. A query like "find discussions about Alice while she was at Beta Corp" is a single Cypher traversal in OpenStinger — it's impossible in any flat-file or chunk-based memory system. As session history grows, OpenStinger's advantage compounds.

---

## Mode 3: Exclusive (Full Commitment)

Disable the framework's native memory entirely. OpenStinger is the only memory backend. This gives the cleanest, most coherent setup — one source of truth, one search interface, one entity graph.

### For OpenClaw

Disable the memory plugin in your OpenClaw config:

```json
{
  "plugins": {
    "slots": {
      "memory": "none"
    }
  }
}
```

> **Warning:** This removes `memory_search` and `memory_get` tools entirely from your OpenClaw agent. Your agent must use OpenStinger MCP tools exclusively. Ensure OpenStinger is fully configured and ingesting sessions before disabling native memory.

### For Nanobot

Remove or disable Nanobot's memory module from its config. Nanobot's philosophy ("small core, external everything") makes this the natural final state — Nanobot handles execution, OpenStinger handles all memory.

### For ZeroClaw

Switch ZeroClaw's memory trait from `sqlite` or `ephemeral` to your external MCP endpoint:

```toml
[memory]
backend = "external"
mcp_server = "openstinger"
```

Or simply configure ZeroClaw to not load any local memory module — all memory operations route to OpenStinger via MCP tool calls.

### For NanoClaw swarms

In exclusive mode, each swarm agent has its own OpenStinger namespace (`agent_namespace` in config). Shared entities (people, organizations, decisions) are recognized consistently across all swarm members via the shared `entity_registry`. No cross-agent memory confusion. No duplicated entity graphs.

```yaml
# Agent A config
agent_namespace: researcher

# Agent B config
agent_namespace: writer

# Same OpenStinger instance — shared entity_registry
# "Alice" = same UUID in both namespaces
```

---

## Migration Path Summary

```
Day 1     → Mode 1 (Alongside)
            Add OpenStinger. Let it ingest 1,000+ episodes.
            Both memory systems active. Zero risk.

Week 1-2  → Mode 2 (Primary)
            Update system prompt. OpenStinger answers first.
            Native memory is the safety net. Still zero risk.

Month 1+  → Mode 3 (Exclusive — optional)
            Disable native memory. Full confidence.
            One source of truth. Maximum coherence.
```

You can stop at Mode 1 or Mode 2 indefinitely. Mode 3 is for when you're confident and want the cleanest setup. There is no requirement to reach Mode 3.

---

## Which Mode for Which Framework?

| Framework | Recommended Start | Notes |
|---|---|---|
| **OpenClaw** | Mode 1 or 2 | QMD + Graphiti are capable; OpenStinger adds governance layer |
| **Nanobot** | Mode 1 → Mode 3 fast | Memory is weak; natural to go exclusive quickly |
| **ZeroClaw** | Mode 2 or 3 immediately | Ephemeral memory = nothing to preserve; OpenStinger fills the gap |
| **NanoClaw** | Mode 3 for swarms | Exclusive + namespaces = coherent swarm memory |
| **PicoClaw** | Mode 3 only | Hardware can't run local DB; OpenStinger on central server is the only viable option |

---

*For framework-specific setup steps, see the individual integration guides:*
- [OpenClaw](OPENCLAW.md)
- [Nanobot](NANOBOT.md)
- [ZeroClaw](ZEROCLAW.md)
- [NanoClaw](NANOCLAW.md)
- [PicoClaw](PICOCLAW.md)
