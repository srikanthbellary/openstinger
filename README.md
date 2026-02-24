# OpenStinger

**Three-tier AI agent memory, reasoning, and alignment harness for OpenClaw.**

OpenStinger gives your OpenClaw agents persistent memory that learns over time, a self-building knowledge vault distilled from their own sessions, and an alignment layer that keeps responses consistent with the agent's stated values and constraints.

---

## What It Does

| Tier | Name | What it adds |
|---|---|---|
| **Tier 1** | Memory Harness | Bi-temporal episodic memory stored in FalkorDB. Conflict-resolving knowledge graph. 9 MCP tools. |
| **Tier 2** | VectraVault | Autonomous classification of sessions into structured self-knowledge notes. 15 MCP tools total. |
| **Tier 3** | Gradient | Synchronous alignment evaluation before every response. Drift detection. Correction engine. 20 MCP tools total. |

You install tiers additively — Tier 1 alone is useful. Add Tier 2 when you want self-building knowledge. Add Tier 3 when you want behavioral alignment and drift detection.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.10 or 3.11 recommended |
| Docker Desktop | For FalkorDB |
| OpenClaw v2026.2+ | Session files at `agents/{name}/sessions/*.jsonl` |
| Novita API key | For LLM inference + embeddings (`https://api.novita.ai/v3/openai`) |

> **Anthropic API key** is optional — the default config uses Novita (DeepSeek V3.2 for inference, qwen3-embedding-8b for embeddings). Set `llm_base_url` in `config.yaml` to switch providers.

---

## Directory Layout (Best Practice)

Keep OpenStinger **next to** OpenClaw — not inside it.

```
C:\Users\you\CLAUDE_CODE\
├── openclaw\
│   └── openclaw-data\
│       ├── config\
│       │   └── agents\main\sessions\   ← sessions OpenStinger reads
│       └── workspace\
│           └── skills\openstinger\      ← SKILL.md for Claudia
└── openstinger\
    ├── .openstinger\                    ← runtime data (git-ignored)
    │   ├── openstinger.db               ← SQLite operational DB
    │   └── vault\                      ← vault notes (Tier 2)
    ├── config.yaml                     ← your config
    ├── .env                            ← API keys
    └── src\openstinger\
```

---

## Installation

### Step 1 — Python environment

```bash
cd C:\Users\you\CLAUDE_CODE\openstinger
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"         # core + dev tools
pip install -e ".[dev,tools]"   # + Datasette SQLite browser (recommended)
```

### Step 2 — Start FalkorDB

```bash
docker compose up -d
docker exec openstinger_falkordb redis-cli ping   # → PONG
```

### Step 3 — Configure

Copy and edit `.env`:
```
# Novita API key (used for both LLM inference and embeddings)
OPENAI_API_KEY=sk_u_...your_novita_key...
```

Key settings in `config.yaml`:
```yaml
agent_name: main
agent_namespace: main

llm:
  provider: openai
  model: deepseek/deepseek-v3.2             # Novita — inference
  fast_model: qwen/qwen3-4b-fp8             # Novita — fast evals (Tier 3)
  llm_base_url: "https://api.novita.ai/v3/openai"
  embedding_model: qwen/qwen3-embedding-8b  # Novita — 1536 dims
  embedding_base_url: "https://api.novita.ai/v3/openai"

ingestion:
  sessions_dir: "C:/Users/you/CLAUDE_CODE/openclaw/openclaw-data/config/agents/main/sessions"
  session_format: openclaw

mcp:
  transport: sse
  tcp_port: 8765
```

### Step 4 — Verify

```bash
python -m pytest tests/test_smoke.py -q
# Expected: 12 passed
```

---

## Starting OpenStinger

### Recommended: startup script (starts everything)

```bash
# Windows
scripts\start.bat               # Tier 1 (memory)
scripts\start.bat tier2         # Tier 2 (memory + vault)
scripts\start.bat tier3         # Tier 3 (memory + vault + alignment)
scripts\start.bat status        # show what's running
scripts\start.bat stop          # stop MCP server + Datasette

# Linux/Mac/Git Bash
./scripts/start.sh              # same options
```

This starts FalkorDB, FalkorDB Browser, the MCP server, and Datasette in one command.

### Manual: MCP server only

```bash
nohup python -m openstinger.mcp.server > /tmp/ov.log 2>&1 &              # Tier 1
nohup python -m openstinger.scaffold.mcp.server > /tmp/ov.log 2>&1 &     # Tier 2
nohup python -m openstinger.gradient.mcp.server > /tmp/ov.log 2>&1 &     # Tier 3
```

The server connects to FalkorDB, initializes the schema, and begins watching your sessions directory automatically. All three entry points use SSE transport via Starlette/Uvicorn.

---

## OpenClaw Integration

### Architecture

```
Windows host
├── OpenStinger MCP server (Python, SSE on port 8765)
│   ├── reads: openclaw-data/config/agents/main/sessions/ (read-only)
│   └── writes: FalkorDB (localhost:6379)
│
Docker containers
├── openclaw-gateway
│   └── mcporter → http://host.docker.internal:8765/sse
└── openstinger_falkordb (port 6379)
```

OpenStinger runs **natively on the Windows host**. OpenClaw runs in Docker and reaches it via `host.docker.internal:8765`.

### Step 1 — Install mcporter in the OpenClaw container

```bash
docker exec openclaw-gateway sh -c "
  mkdir -p /home/node/.npm-global &&
  npm config set prefix '/home/node/.npm-global' &&
  npm install -g mcporter
"
```

### Step 2 — Add the OpenStinger server to mcporter

```bash
docker exec -i openclaw-gateway sh << 'EOF'
export PATH=/home/node/.npm-global/bin:$PATH
mcporter config add openstinger http://host.docker.internal:8765/sse \
  --config /app/config/mcporter.json
EOF
```

### Step 3 — Install the SKILL.md for Claudia

```bash
mkdir -p openclaw-data/workspace/skills/openstinger
# Copy SKILL.md from this repo (see openclaw-data/workspace/skills/openstinger/SKILL.md)
```

The SKILL.md is already present if you cloned this repo alongside OpenClaw. OpenClaw's skill watcher picks it up automatically — no restart needed.

### Step 4 — Start OpenStinger and verify

```bash
# On the Windows host:
nohup python -m openstinger.mcp.server > /tmp/ov.log 2>&1 &

# Inside OpenClaw container:
docker exec -i openclaw-gateway sh << 'EOF'
export PATH=/home/node/.npm-global/bin:$PATH
mcporter list --config /app/config/mcporter.json
mcporter call openstinger.memory_namespace_status \
  --args '{"agent_namespace":"main"}' --config /app/config/mcporter.json
EOF
```

Expected: `9 tools, 1 server healthy` and namespace stats.

---

## Available Tools

### Tier 1 — 9 tools

| Tool | Description |
|---|---|
| `memory_add` | Store an episode manually |
| `memory_query` | Hybrid BM25 + vector search |
| `memory_search` | BM25 keyword search (episodes/entities/facts) |
| `memory_get_entity` | Fetch entity + relationships by UUID |
| `memory_get_episode` | Fetch episode by UUID |
| `memory_job_status` | Check ingestion job status |
| `memory_ingest_now` | Trigger immediate backlog ingestion (fire-and-forget) |
| `memory_namespace_status` | Health stats: episode/entity/edge counts |
| `memory_list_agents` | List all registered namespaces |

### Tier 2 adds 6 tools (15 total)

`vault_status`, `vault_sync_now`, `vault_stats`, `vault_promote_now`, `vault_note_list`, `vault_note_get`

### Tier 3 adds 5 tools (20 total)

`gradient_status`, `gradient_alignment_score`, `gradient_drift_status`, `gradient_alignment_log`, `gradient_alert`

---

## Upgrade Path

```
Tier 1 → run Tier 1 server (start here)
   ↓  (~1,000+ episodes ingested)
Tier 2 → run Tier 2 server (vault classification begins)
   ↓  (~1-2 weeks, vault has identity notes)
Tier 3 → run Tier 3 server (observe_only=true by default — safe)
```

See `docs/07_CROSS_TIER_INSTALL_SEQUENCE.md` for verification gates between upgrades.

---

## Session Format

OpenStinger parses OpenClaw v3 JSONL format automatically (`session_format: openclaw`). It extracts:
- **User messages** → `source: openclaw_user`
- **Assistant responses** → `source: openclaw_assistant`

Skipped: thinking blocks, tool calls, session metadata.

For non-OpenClaw JSONL: use `session_format: simple` with `{"content": "...", "source": "...", "valid_at": 1234567890}`.

---

## Multi-Agent Setup

Run separate server instances with different `config.yaml` files for each agent:

```yaml
# config-dev.yaml
agent_name: dev
agent_namespace: dev
ingestion:
  sessions_dir: ".../openclaw-data/config/agents/dev/sessions"
```

```bash
nohup python -m openstinger.mcp.server --config config-dev.yaml > /tmp/ov-dev.log 2>&1 &
```

---

## Data Locations

| What | Where | Owned by |
|---|---|---|
| Session files | `openclaw-data/config/agents/*/sessions/` | OpenClaw (read-only) |
| SQLite DB | `.openstinger/openstinger.db` | OpenStinger |
| Vault notes | `.openstinger/vault/` | OpenStinger |
| FalkorDB graphs | Docker volume `openstinger_falkordb_data` | FalkorDB |
| mcporter config | `openclaw-data/config/mcporter.json` (inside container: `/app/config/mcporter.json`) | OpenStinger |
| Claudia SKILL.md | `openclaw-data/workspace/skills/openstinger/SKILL.md` | OpenStinger |

---

## Browser Verification UIs

Two browser tools are included to inspect and validate the memory state visually:

| URL | Tool | What it shows |
|---|---|---|
| `http://localhost:3000` | FalkorDB Browser | Knowledge graph — entities, facts, relationships (visual) |
| `http://localhost:8001` | Datasette | SQLite operational DB — all 11 tables |

**FalkorDB Browser** starts automatically with `docker compose up -d` (included in `docker-compose.yml`). Login with host `host.docker.internal`, port `6379`, no password — then select graph `openstinger_temporal`.

**Datasette** requires the `[tools]` extra: `pip install -e ".[tools]"`. Starts with `scripts/start.sh` or manually: `datasette .openstinger/openstinger.db --port 8001`.

See `VERIFICATION_GUIDE.md` for full validation queries and a health checklist.

---

## Troubleshooting

**FalkorDB not connecting**
```bash
docker compose up -d
docker exec openstinger_falkordb redis-cli ping   # → PONG
```

**Port 8765 already in use**
```bash
netstat -ano | grep :8765       # find PID
powershell -Command "Stop-Process -Id <PID> -Force"
```

**No episodes ingesting**
1. Check `sessions_dir` in `config.yaml` — must be Windows host path (forward slashes)
2. Verify `.jsonl` files exist in that directory
3. Confirm `session_format: openclaw`
4. Check log: `tail -f /tmp/ov.log`

**mcporter: openstinger appears offline**
The MCP server is not running. Start it on the Windows host first.

**Empty search results**
If `episode_count` is low, run `memory_ingest_now` to trigger backlog ingestion.
If episode_count is healthy but search returns nothing, queries may be too specific — try broader terms.

**"Profile state: empty" in Gradient**
Normal for new deployments — Tier 2 must classify sessions before Tier 3 builds a profile. See `docs/09_ALIGNMENT_PROFILE_BOOTSTRAPPING_GUIDE.md`.

---

## Reference Documentation

| Document | What it covers |
|---|---|
| `docs/01_DEPENDENCY_MANIFEST.md` | Exact package versions, platform notes |
| `docs/02_GRAPHITI_FORK_DIFF.md` | Graphiti v0.24.0 fork changes |
| `docs/03_ALGORITHM_REFERENCE.md` | Bi-temporal model, deduplication, namespace isolation |
| `docs/04_FALKORDB_CYPHER_REFERENCE.md` | All FalkorDB queries, verified dialect |
| `docs/05_OPERATIONAL_DB_SCHEMA.md` | 11-table schema across all 3 tiers |
| `docs/06_INTEGRATION_TEST_SCENARIOS_TIER1.md` | 35 integration test scenarios |
| `docs/07_CROSS_TIER_INSTALL_SEQUENCE.md` | Install gates, verification, rollback |
| `docs/08_SCAFFOLD_CLASSIFICATION_PROMPT_REFERENCE.md` | Vault classification prompts |
| `docs/09_ALIGNMENT_PROFILE_BOOTSTRAPPING_GUIDE.md` | Tier 3 profile bootstrap |
| `docs/10_GRADIENT_PROMPT_ENGINEERING_REFERENCE.md` | Evaluation pipeline prompts |
| `docs/11_MCP_TOOL_ROUTING_REFERENCE.md` | 20-tool routing table |
| `docs/12_SKILL_FILE_REFERENCE.md` | Agent skill files |
| `docs/13_OBSERVE_MODE_CALIBRATION_GUIDE.md` | Gradient calibration protocol |

---

## License

MIT
