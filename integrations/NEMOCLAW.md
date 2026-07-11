# NemoClaw Integration Guide

**OpenStinger version:** v0.8+  
**NemoClaw version:** Alpha (March 2026)  
**Integration effort:** Configuration only — no code changes required  
**Session format:** `openclaw` (NemoClaw runs standard OpenClaw underneath)

> **Status note:** NemoClaw is alpha software announced at NVIDIA GTC on March 16, 2026. APIs and behavior may change. This guide is verified against the alpha release. Check [github.com/NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw) for upstream changes before deploying.

---

## 1. Overview

NemoClaw is NVIDIA's open-source wrapper that installs OpenClaw inside an [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandbox — adding kernel-level process isolation, policy-governed network egress, and Nemotron model defaults on top of a standard OpenClaw install.

**NemoClaw does not modify OpenClaw's internals.** The agent running inside the sandbox is standard OpenClaw. It writes the same JSONL session files, exposes the same MCP interface, and behaves identically at the application layer. OpenStinger's `openclaw` session format works without modification.

What NemoClaw adds are two infrastructure constraints that require deliberate configuration for OpenStinger to work correctly:

| Constraint | Impact on OpenStinger | Resolution |
|---|---|---|
| **Filesystem isolation** | Session files and profile dirs land inside the sandbox at `/sandbox/`, not on the host at `~/.openclaw/` | Declare a host volume mount in the OpenShell blueprint before sandbox creation |
| **Network egress policy** | All outbound connections blocked by default — including the MCP SSE call to `localhost:8766` | Add `localhost:8766` to the egress allowlist in the blueprint YAML |

Both are one-time configuration steps. No OpenStinger code changes are needed.

---

## 2. Prerequisites

- **OpenStinger v0.8+** installed and running on the host
- **NemoClaw alpha** installed: `git clone https://github.com/NVIDIA/NemoClaw && cd NemoClaw && ./install.sh`
- **NVIDIA OpenShell** installed (NemoClaw's `install.sh` handles this)
- **Linux Ubuntu 22.04 LTS or later** (NemoClaw's current OS requirement)
- **Docker** installed and running on the host
- OpenStinger's three services running on the host before the sandbox starts:
  - FalkorDB: `docker compose up -d`
  - Tier 1 MCP server: `python -m openstinger.mcp.server` (port 8766)
  - Tier 2 (optional): `python -m openstinger.scaffold.mcp.server`
  - Tier 3 (optional): `python -m openstinger.gradient.mcp.server`

> **Important:** OpenStinger runs on the **host**, not inside the NemoClaw sandbox. FalkorDB, PostgreSQL, and the MCP server are all host-side services. The OpenClaw agent inside the sandbox connects to them over the network.

---

## 3. Install

No additional OpenStinger packages are required for NemoClaw. The `openclaw` session format is built into OpenStinger core and has been since v0.1.

```bash
# Verify your OpenStinger version
openstinger-cli --version
# Should report v0.8 or later

# Verify MCP server is reachable
curl -s http://localhost:8766/sse --max-time 2
# Should return an SSE stream header
```

---

## 4. Configure the OpenShell Blueprint

This is the most important step. NemoClaw's sandbox is created from a **blueprint** — a versioned Python artifact that declares the sandbox's filesystem mounts, network policy, and inference configuration. The blueprint is applied at sandbox creation time. Filesystem policy is **locked** at creation and cannot be changed without recreating the sandbox.

### 4.1 Locate the Blueprint

NemoClaw's default blueprint lives in the `nemoclaw-blueprint/` directory of the cloned repo. The primary configuration file is `nemoclaw-blueprint/config.yaml` (exact path may vary — run `nemoclaw setup --dry-run` to confirm).

### 4.2 Declare the Session Files Volume Mount

Add a volume mount so OpenStinger can read session files from the host side. The sandbox writes OpenClaw sessions to `/sandbox/.openclaw/agents/main/sessions/` inside the container. Map this to a host path:

```yaml
# nemoclaw-blueprint/config.yaml — add to the volumes section

volumes:
  # OpenStinger session file bridge
  # Maps sandbox OpenClaw sessions to a host path OpenStinger can read
  - host: ~/.openstinger/nemoclaw-sessions
    sandbox: /sandbox/.openclaw/agents/main/sessions
    mode: read-write

  # If you use the AgentProfileIngester (v0.7+), also mount the workspace:
  - host: ~/.openstinger/nemoclaw-workspace
    sandbox: /sandbox/.openclaw/workspace
    mode: read-only
```

Create the host directories before running the installer:

```bash
mkdir -p ~/.openstinger/nemoclaw-sessions
mkdir -p ~/.openstinger/nemoclaw-workspace
```

### 4.3 Declare the Network Egress Allowlist

Add OpenStinger's MCP server to the sandbox's allowed outbound hosts. NemoClaw's network policy blocks all egress by default and surfaces blocked requests in the OpenShell TUI for operator approval. Adding it to the allowlist upfront makes the integration reproducible:

```yaml
# nemoclaw-blueprint/config.yaml — add to the network section

network:
  egress:
    allowlist:
      # OpenStinger Tier 1 memory harness
      - host: localhost
        port: 8766
        description: "OpenStinger MCP server — Tier 1 (memory)"

      # Add these if you run Tier 2 and/or Tier 3:
      # - host: localhost
      #   port: 8767
      #   description: "OpenStinger Scaffold MCP server — Tier 2 (vault + knowledge)"
      # - host: localhost
      #   port: 8768
      #   description: "OpenStinger Gradient MCP server — Tier 3 (alignment)"
```

> **If you skip this step:** The first time your OpenClaw agent inside the sandbox calls an OpenStinger tool, OpenShell will block the request and show it in the `nemoclaw term` TUI. Approve it there and it becomes permanently allowed for this sandbox — equivalent to the allowlist declaration but manual. Both paths work. The YAML approach is reproducible across sandbox recreations.

---

## 5. Configure OpenStinger

Point OpenStinger's ingestion at the host-side session directory declared in the blueprint mount:

```yaml
# ~/.openstinger/config.yaml

openclaw:
  agent_id: main
  sessions_dir: ~/.openstinger/nemoclaw-sessions   # matches the host path in the volume mount

ingestion:
  session_format: openclaw     # NemoClaw runs standard OpenClaw — no format change needed
  poll_interval_seconds: 30
  concurrency: 5

# AgentProfileIngester (v0.7+) — point at the mounted workspace
profile_dirs:
  - ~/.openstinger/nemoclaw-workspace
```

Full config reference: see `config.yaml.example` in the OpenStinger repo root.

---

## 6. Connect OpenStinger to the OpenClaw Agent

Inside the NemoClaw sandbox, the OpenClaw agent is configured via its standard `mcporter` or `openclaw.json` MCP config. Add OpenStinger's MCP server to the agent's tool list:

```json
{
  "mcpServers": {
    "openstinger": {
      "baseUrl": "http://localhost:8766/sse"
    }
  }
}
```

In a NemoClaw environment, this config lives inside the sandbox. Connect to the sandbox and edit it there:

```bash
# Connect to the running sandbox
nemoclaw my-assistant connect

# Inside the sandbox shell:
sandbox@my-assistant:~$ cat ~/.openclaw/mcpServers.json
# Add the openstinger block to the existing config, then:
sandbox@my-assistant:~$ openclaw agent --agent main --local -m "list your available MCP tools" --session-id verify
```

The agent should report OpenStinger's tools in its response. If it doesn't, check the network policy step in Section 4.3.

---

## 7. Drop In the Agent Skill File

Copy OpenStinger's SKILL.md into the OpenClaw workspace inside the sandbox so the agent knows how to use OpenStinger's tools effectively:

```bash
# From the host, copy into the mounted workspace directory
cp /path/to/openstinger/AGENT_SKILL_TEMPLATE.md \
   ~/.openstinger/nemoclaw-workspace/skills/openstinger/SKILL.md
```

Because the workspace is mounted into the sandbox (declared in Section 4.2), this file will be visible to OpenClaw at `~/.openclaw/workspace/skills/openstinger/SKILL.md` inside the sandbox without needing to reconnect.

---

## 8. Verify Ingestion

After starting a session with your NemoClaw agent, verify that OpenStinger is picking up the session files:

```bash
# On the host — check ingestion progress
openstinger-cli progress

# You should see the nemoclaw-sessions path being monitored
# and episode counts incrementing as the session runs

# Spot check with a memory query
openstinger-cli query "what did we just discuss"
```

If `openstinger-cli progress` shows no files being scanned, verify:
1. The volume mount path in the blueprint matches `~/.openstinger/nemoclaw-sessions` exactly
2. The sandbox was created (or recreated) **after** the blueprint edit — filesystem policy is locked at creation
3. The `sessions_dir` in `~/.openstinger/config.yaml` matches the host-side mount path

---

## 9. Three Adoption Modes

OpenStinger supports three adoption modes alongside NemoClaw. All three work the same as with base OpenClaw — NemoClaw's sandbox does not change this.

### Alongside Mode (default)

OpenStinger and NemoClaw's native OpenClaw memory both active. The agent has all tools from both systems. Start here.

```yaml
# config.yaml — no special config needed
# OpenStinger tools available, OpenClaw native memory also active
```

### Primary Mode

OpenStinger is the primary memory backend. OpenClaw's native memory is still active but the agent is instructed (via SKILL.md) to prefer OpenStinger tools for all memory operations.

```yaml
# config.yaml
gradient:
  mode: observe   # Gradient watches but does not intercept — useful starting point
```

Update the SKILL.md inside the sandbox to instruct the agent to use `memory_query` before relying on OpenClaw's native memory.

### Exclusive Mode

OpenStinger is the only memory backend. Disable OpenClaw's native memory in `openclaw.json` (inside the sandbox) and use only OpenStinger tools.

```json
// ~/.openclaw/openclaw.json inside the sandbox
{
  "memory": {
    "enabled": false
  }
}
```

> No migration required between modes. OpenStinger's graph accumulates regardless of which mode you're in — switching to Exclusive later does not lose previously ingested episodes.

---

## 10. Inference and Embedding Considerations

NemoClaw's OpenShell intercepts all model API calls and routes them to the configured inference provider (Nemotron cloud, local NIM, or vLLM). This affects the **agent's LLM calls** — the calls the OpenClaw agent makes for reasoning and generation.

OpenStinger's own LLM calls (entity extraction, embedding, conflict detection) go directly from the host to your configured provider. They bypass the OpenShell inference gateway entirely because OpenStinger runs on the host, not inside the sandbox.

**Your OpenStinger `config.yaml` LLM config is independent of NemoClaw's inference profile.** Configure them separately:

```yaml
# ~/.openstinger/config.yaml — OpenStinger's own LLM config
# This is independent of NemoClaw's Nemotron inference profile

llm:
  provider: openai
  model: claude-sonnet-4-6
  llm_base_url: "https://api.anthropic.com/v1"

embedding_provider: ollama          # v0.8 — Ollama local embeddings
ollama_host: "http://localhost:11434"
embedding_model: nomic-embed-text
vector_dimensions: 768
```

If you want to use Nemotron for OpenStinger's entity extraction (i.e., route OpenStinger's LLM calls through NIM), point `llm_base_url` at your local NIM endpoint and set the model accordingly. This is optional and not required for the integration to work.

---

## 11. Troubleshooting

**OpenStinger MCP tools not appearing in the agent's tool list**

The agent can't reach `localhost:8766`. Check:
- OpenStinger MCP server is running on the host: `curl -s http://localhost:8766/sse --max-time 2`
- `localhost:8766` is in the OpenShell egress allowlist (Section 4.3)
- If you skipped the allowlist step, run `nemoclaw term` to open the OpenShell TUI and approve the pending blocked connection

**`openstinger-cli progress` shows no session files**

The volume mount isn't working. Check:
- The blueprint was edited before sandbox creation — not after. The filesystem policy locks at creation time.
- Recreate the sandbox: `nemoclaw my-assistant stop && nemoclaw setup` after editing the blueprint
- Verify the host directory exists: `ls ~/.openstinger/nemoclaw-sessions/`
- Verify the mount inside the sandbox: `nemoclaw my-assistant connect` then `ls /sandbox/.openclaw/agents/main/sessions/`

**Sessions directory exists but no JSONL files appear**

The session has not started yet, or OpenClaw inside the sandbox is writing to a different path. Check which agent ID OpenClaw is using inside the sandbox:

```bash
nemoclaw my-assistant connect
sandbox@my-assistant:~$ openclaw agent --agent main --local -m "what is your agent ID" --session-id check
```

Adjust the `sessions_dir` mount in the blueprint to match the actual agent ID path if it differs from `main`.

**Ingestion working but memory queries return nothing**

The ingestion scheduler may be running but the FalkorDB embedding is failing. Check:
- `openstinger-cli progress` — are episodes being written to FalkorDB or failing?
- `docker logs falkordb` — any connection errors?
- Verify your `OPENAI_API_KEY` (or Ollama config) is set correctly in `~/.openstinger/.env`

**NemoClaw blueprint schema has changed since this guide was written**

NemoClaw is alpha software. If the blueprint YAML structure has changed, check [github.com/NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw) for the current schema and open an issue in the OpenStinger repo if this guide needs updating.

---

## 12. What Is Not Supported (Yet)

- **UNIX socket MCP transport:** OpenStinger currently uses SSE over HTTP. A UNIX socket transport would bypass the network policy entirely — planned consideration for v1.0, not NemoClaw-specific.
- **Nemotron as OpenStinger's embedding provider:** Nemotron is not an embedding model. OpenStinger's embedding must use a separate provider (OpenAI, Ollama, etc.) even if the agent uses Nemotron for generation.
- **Gradient interceptor for NemoClaw's inference pipeline:** OpenStinger-Gradient operates at the application output layer, not at the inference routing layer. Integrating Gradient with OpenShell's inference intercept would require middleware work — not scoped for the current release.
- **Multi-sandbox namespace isolation:** Running multiple NemoClaw sandboxes with separate OpenStinger namespaces is architecturally possible (each sandbox gets its own `sessions_dir` mount and OpenStinger namespace config), but is not documented or tested. Raise a GitHub issue if this is your use case.

---

## See Also

- `integrations/OPENCLAW.md` — Base OpenClaw integration (NemoClaw's foundation)
- `integrations/INTEGRATION_MODES.md` — Alongside / Primary / Exclusive mode reference
- `AGENT_SKILL_TEMPLATE.md` — Full skill file for injecting OpenStinger tool guidance
- [NVIDIA NemoClaw GitHub](https://github.com/NVIDIA/NemoClaw) — upstream source
- [NVIDIA OpenShell docs](https://github.com/NVIDIA/OpenShell) — sandbox policy reference
