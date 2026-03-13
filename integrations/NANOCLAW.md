# OpenStinger + NanoClaw

NanoClaw is built on Anthropic's Agent SDK and focuses on agent swarms — multiple coordinated agents running in parallel. Swarms need shared memory and knowledge coordination. OpenStinger's multi-agent namespace architecture was designed for exactly this.

## What OpenStinger adds to NanoClaw

| Without OpenStinger | With OpenStinger |
|---|---|
| Each swarm agent memory-isolated | Shared entity registry across all agents |
| No cross-agent knowledge | Entities recognized identically across namespaces |
| No governance layer | Agent registry with parent/child hierarchy |
| No audit trail | Full alignment event log per agent |

**The swarm story:** NanoClaw introduces agent swarms. OpenStinger gives those swarms a coherent shared memory. NanoClaw solves execution isolation; OpenStinger solves knowledge coherence.

## Setup

### 1. Start OpenStinger

```bash
python -m openstinger.gradient.mcp.server
```

### 2. Add OpenStinger to NanoClaw's MCP config

NanoClaw uses Anthropic's Agent SDK (MCP-native). In your MCP server config:

```json
{
  "mcpServers": {
    "openstinger": {
      "url": "http://localhost:8766/sse"
    }
  }
}
```

### 3. Configure agent namespaces

Each swarm agent gets its own OpenStinger namespace. In `config.yaml`:

```yaml
agent_namespace: researcher    # or orchestrator, writer, analyst  one per agent
```

The same OpenStinger instance serves all namespaces. Shared entities (people, projects, decisions) are recognized consistently across all agents.

## Multi-Agent Memory Pattern

```
Orchestrator (namespace: main)
 reads: all shared entities
 writes: cross-agent decisions

Research Agent (namespace: research)  Writing Agent (namespace: writing)
 reads: own episodes + entities     reads: own episodes + entities
 writes: own episodes               writes: own episodes

Shared entity_registry: "Alice" = same UUID in all namespaces
```

## About Tier 3: Gradient (Per-Agent Alignment)

OpenStinger's Tier 3 runs alignment evaluation per-response using:

> **dE/dt = β(C–D)E**

**E** = accumulated episodic memory · **β** = distilled self-knowledge · **C** = the agent's own defined constraints · **D** = observed deviation. In a NanoClaw swarm, each agent runs its own Gradient evaluation against its own namespace's knowledge — alignment is measured per-agent, not globally. The orchestrator can query `gradient_history` and `drift_status` across all agents to detect which swarm member is drifting.

---

## Integration Mode

→ **[Full integration modes guide with config snippets](INTEGRATION_MODES.md)**

**Recommended for NanoClaw:** Use **Mode 3 (Exclusive)** across all swarm agents with separate namespaces. This gives each agent isolated episodic memory while sharing the `entity_registry` — the cleanest architecture for coherent swarm coordination.

---

*For the full OpenStinger setup guide, see the [main README](../README.md).*
