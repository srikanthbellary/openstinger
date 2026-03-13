# OpenStinger + PicoClaw

PicoClaw is built for edge hardware  routers, cameras, microcontrollers  with a target footprint under 10MB RAM. Local graph databases are architecturally impossible on this hardware. OpenStinger runs on a central server and gives every PicoClaw device persistent, queryable memory without touching the device's footprint.

## What OpenStinger adds to PicoClaw

PicoClaw's hardware constraints mean memory will always be flat files or minimal SQLite on-device. OpenStinger offloads all memory operations to a server:

| PicoClaw alone | PicoClaw + OpenStinger (central) |
|---|---|
| Flat file or minimal SQLite | Full bi-temporal knowledge graph |
| Memory limited by device RAM | Memory limited only by server storage |
| No cross-device entity awareness | Fleet-wide shared entity registry |
| No audit trail | Complete operational event log |

**The fleet story:** Run PicoClaw on \ edge devices. Run OpenStinger on a central server. Multiple devices share entity awareness  they collectively build a shared knowledge graph of whatever environment they're monitoring or operating in.

## Setup

### 1. Start OpenStinger on a central server

OpenStinger runs on any machine with Python 3.10+ and Docker. Edge devices connect over the network.

```bash
python -m openstinger.gradient.mcp.server
```

Expose port 8766 on your central server. Configure firewall as appropriate.

### 2. Connect each PicoClaw device

In each device's MCP client config, point to the central OpenStinger server:

```json
{
  "mcpServers": {
    "openstinger": {
      "url": "http://your-server-ip:8766/sse"
    }
  }
}
```

### 3. Assign each device its own namespace

In `config.yaml` for each device's OpenStinger session ingestion:

```yaml
agent_namespace: device-001   # unique per device
```

Shared entities (locations, devices, events) appear in the shared `entity_registry`. Each device's episodic memory stays isolated in its own namespace graph.

## About Tier 3: Gradient

OpenStinger's Tier 3 (Gradient) runs alignment evaluation per-response on the central server — not on the edge device. PicoClaw agents call `gradient_alignment_score` or `ops_status` via MCP; all computation happens server-side.

> **dE/dt = β(C–D)E** — every response scored against the agent's own evolving baseline. Zero compute on the edge device.

---

## Integration Mode

→ **[Full integration modes guide with config snippets](INTEGRATION_MODES.md)**

**Recommended for PicoClaw:** **Mode 3 (Exclusive)** is the only viable option. Hardware constraints prevent any local database — OpenStinger on a central server IS the memory system. For a fleet of devices, a single OpenStinger instance with per-device namespaces serves all devices simultaneously.

---

*For the full OpenStinger setup guide, see the [main README](../README.md).*
