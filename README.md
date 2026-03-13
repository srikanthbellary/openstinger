# OpenStinger

<p align="center">
  <img src="assets/OpenStinger_Logo_v3_transparent.png" alt="OpenStinger" width="480">
</p>

<p align="center">
  <strong>One memory layer. Every agent framework.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python 3.10+"></a>
  <a href="https://falkordb.com"><img src="https://img.shields.io/badge/FalkorDB-1.6%2B-orange?style=for-the-badge" alt="FalkorDB"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-1.26%2B-green?style=for-the-badge" alt="MCP"></a>
</p>

<p align="center">
  Works with&nbsp;
  <strong>OpenClaw &nbsp;·&nbsp; Nanobot &nbsp;·&nbsp; ZeroClaw &nbsp;·&nbsp; NanoClaw &nbsp;·&nbsp; PicoClaw &nbsp;·&nbsp; Claude Code &nbsp;·&nbsp; Cursor &nbsp;·&nbsp; Qwen-Agent &nbsp;·&nbsp; DeerFlow &nbsp;·&nbsp; LangGraph</strong><br>
  — any MCP-compatible runtime. One endpoint. Zero lock-in.
</p>

---

The autonomous era needs more than execution. Agents hallucinate facts. They drift from their values. They forget who they are. **OpenStinger** is the memory, reasoning, and alignment infrastructure that keeps them grounded — exposed as 30 MCP tools any agent calls natively.

Built on [FalkorDB](https://falkordb.com) (bi-temporal graph + vector) and [PostgreSQL](https://postgresql.org) (operational audit DB), served over [Model Context Protocol](https://modelcontextprotocol.io). No SDK changes. No vendor lock-in.

## Integration Modes

OpenStinger works alongside any *Claw framework's built-in memory — or replaces it entirely. Three adoption modes, zero switching cost:

| Mode | How | When |
|---|---|---|
| **Alongside** | Both OpenStinger + native memory active. Agent has all tools. | Day 1 — add OpenStinger, change nothing |
| **Primary** | System prompt: query OpenStinger first, fall back to native if empty | After ~1,000 episodes are ingested |
| **Exclusive** | Disable native memory. OpenStinger is the only backend. | Full commitment — cleanest setup |

Start alongside. Go primary when ready. No framework migration required.

→ **[Full integration modes guide with per-framework configs](integrations/INTEGRATION_MODES.md)**

---

## Compatible Frameworks

| Framework | MCP | Integration Guide |
|---|---|---|
| **OpenClaw** | ✅ via mcporter | [View guide](integrations/OPENCLAW.md) |
| **DeerFlow** | ✅ v0.7 adapter | [View guide](integrations/DEERFLOW.md) |
| **Qwen-Agent** | ✅ v0.7 adapter | [View guide](integrations/QWEN_AGENT.md) |
| **LangGraph** | ✅ MCP native | [View guide](integrations/LANGGRAPH_INTEGRATION_GUIDE.md) |
| **Nanobot** | ✅ confirmed Feb 2026 | [View guide](integrations/NANOBOT.md) |
| **ZeroClaw** | ✅ swappable trait | [View guide](integrations/ZEROCLAW.md) |
| **NanoClaw** | ✅ Agent SDK native | [View guide](integrations/NANOCLAW.md) |
| **PicoClaw** | ✅ v0.8 | [View guide](integrations/PICOCLAW.md) |
| **Claude Code** | ✅ MCP native | Point at `http://localhost:8766/sse` |
| **Cursor** | ✅ MCP native | Point at `http://localhost:8766/sse` |

---

## What It Does

Three additive tiers. Start with Tier 1 and unlock the rest as data accumulates.

| Tier | Name | Tools | What it gives your agent |
|---|---|---|---|
| **Tier 1** | Memory Harness | 11 | Bi-temporal episodic memory. Every session ingested automatically. Hybrid BM25 + vector semantic search. Date filtering. Numeric/IP search. Delete and update stored memories. |
| **Tier 2** | StingerVault | 11 | Autonomous distillation of sessions into structured self-knowledge: identity, domain, methodology, preferences, constraints. External document ingestion (URL, PDF, YouTube). |
| **Tier 3** | Gradient | 8 | Synchronous alignment evaluation before every response. Drift detection. Correction engine. 3 new observability tools (v0.7). Starts in observe-only mode. |

> ⚡ **Tier 3 ships in `observe_only` mode.**
> It evaluates every response and logs alignment scores — but never blocks or corrects anything until you switch it on.
> This is intentional: let it calibrate against real sessions first.
> Run `openstinger-cli progress` to see readiness, then set `observe_only: false` in `config.yaml` and restart.

---

## Quick Start (5 minutes)

**Requirements:** Python 3.10+, Docker Desktop, one API key (Anthropic or any OpenAI-compatible provider)

### 1. Clone and install
```bash
git clone https://github.com/srikanthbellary/openstinger.git
cd openstinger
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (Git Bash)
source .venv/Scripts/activate
# Windows (cmd / PowerShell)
.venv\Scripts\activate

pip install -e "."
```

### 2. Configure
```bash
cp .env.example .env
cp config.yaml.example config.yaml
```

Edit `.env` — add your API keys:
```env
# Anthropic (for LLM — entity extraction, dedup, conflict resolution)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (for embeddings) — OR use any OpenAI-compatible provider
OPENAI_API_KEY=sk-...

# FalkorDB password — leave blank for local dev (simplest)
FALKORDB_PASSWORD=

# PostgreSQL password (used by docker compose)
POSTGRES_PASSWORD=your_postgres_password
```

> **⚠️ Password gotcha:** If you set `FALKORDB_PASSWORD`, do NOT use `#` in it.
> `.env` files treat `#` as a comment — `myPass#2026` becomes `myPass`.
> Use only alphanumeric characters: `myPass2026` is safe.

Edit `config.yaml` — the only required change:
```yaml
ingestion:
  sessions_dir: "/path/to/your/agent/sessions"   # path to your agent's JSONL session files
```

### 3. Start FalkorDB
```bash
docker compose up -d
```

### 4. Start OpenStinger

**Tier 1 only** (memory, 11 tools):
```bash
python -m openstinger.mcp.server
```

**All tiers** (memory + vault + alignment, 30 tools):
```bash
python -m openstinger.gradient.mcp.server
```

### 5. Connect your agent

OpenStinger speaks MCP over SSE. The config is identical across all supported frameworks:

```json
{
  "mcpServers": {
    "openstinger": {
      "baseUrl": "http://localhost:8766/sse"
    }
  }
}
```

> See [integrations/](integrations/) for framework-specific setup guides (OpenClaw, DeerFlow, Qwen-Agent, LangGraph, Nanobot, ZeroClaw, NanoClaw, PicoClaw).

Then your agent can call:
```bash
mcporter call openstinger.memory_query \
  --args '{"query": "what did we work on last week", "limit": 5}'
```

---

## Using OpenAI-Compatible Providers (Novita, DeepSeek, etc.)

OpenStinger works with any OpenAI-compatible API for both LLM and embeddings — no OpenAI account required.

```yaml
# config.yaml
llm:
  provider: openai
  model: deepseek/deepseek-v3.2
  llm_base_url: "https://api.novita.ai/v3/openai"
  embedding_model: qwen/qwen3-embedding-8b
  embedding_provider: openai
  embedding_base_url: "https://api.novita.ai/v3/openai"
```

Set `OPENAI_API_KEY` in `.env` to your Novita (or other) API key.

---

## Local Embeddings via Ollama (v0.8)

Run embeddings entirely on-device — no cloud API key, no network calls, no cost per token. Full local operation for air-gapped environments or privacy-sensitive deployments.

**Supported models:**

| Model | Dimensions | Pull command |
|---|---|---|
| `nomic-embed-text` | 768 | `ollama pull nomic-embed-text` |
| `mxbai-embed-large` | 1024 | `ollama pull mxbai-embed-large` |
| `all-minilm` | 384 | `ollama pull all-minilm` |

**Setup via interactive init (recommended):**

```bash
openstinger-cli init
# → When prompted for "Embedding provider", select "ollama"
# → Select model — dimensions are auto-populated
```

**Or configure manually in `config.yaml`:**

```yaml
llm:
  embedding_provider: ollama
  embedding_model: nomic-embed-text
  ollama_host: http://localhost:11434   # default; change if Ollama runs elsewhere

falkordb:
  vector_dimensions: 768   # must match the model's native output size
```

In `.env`, set `OPENAI_API_KEY=ollama` (the client requires a non-empty value; Ollama ignores it):

```env
OPENAI_API_KEY=ollama
```

> **Breaking change warning:** `vector_dimensions` must match the embedding model's native output size. If you switch models after data has been ingested, you must wipe and re-create the FalkorDB vector indices:
> ```bash
> docker compose down -v && docker compose up -d
> ```
> Then re-ingest your sessions. Embeddings from different dimension sizes are not compatible.

---

## MCP Tools

### Tier 1 — Memory (11 tools)

| Tool | What it does |
|---|---|
| `memory_add` | Store an episode manually |
| `memory_query` | Hybrid BM25 + vector search. Returns unified ranked results. Supports `after_date` / `before_date`. |
| `memory_search` | Smart keyword search with automatic fallbacks: numeric/IP detection, temporal queries, fuzzy entity matching |
| `memory_get_entity` | Fetch an entity and its current relationships by UUID |
| `memory_get_episode` | Fetch a specific episode by UUID |
| `memory_job_status` | Check ingestion job status |
| `memory_ingest_now` | Trigger immediate session ingestion |
| `memory_namespace_status` | Health stats: episode / entity / edge counts |
| `memory_list_agents` | List all registered agent namespaces |
| `memory_delete` | Permanently delete an episode and prune orphaned entities (v0.8) |
| `memory_update` | Update episode content and re-index with new embedding (v0.8) |

### Tier 2 — StingerVault (11 tools)

`vault_status` · `vault_sync_now` · `vault_stats` · `vault_promote_now` · `vault_note_list` · `vault_note_get` · `knowledge_ingest` · `namespace_create` · `namespace_list` · `namespace_archive`

### Tier 3 — Gradient (8 tools)

| Tool | What it does |
|---|---|
| `gradient_status` | Gradient health, profile state, observe_only flag |
| `gradient_alignment_score` | Evaluate a response — returns score + verdict |
| `gradient_drift_status` | Rolling window alignment stats |
| `gradient_alignment_log` | Recent alignment evaluation log |
| `gradient_alert` | Current drift alert status |
| `ops_status` ⭐ | Single-call dashboard: vault notes + gradient pass rate + drift state |
| `gradient_history` ⭐ | Last N alignment verdicts with scores from PostgreSQL |
| `drift_status` ⭐ | Behavioral window drift history from PostgreSQL |

**`ops_status` — what one call returns:**
```json
{
  "vault": {
    "total_notes": 42,
    "active_notes": 38,
    "stale_notes": 4,
    "categories": ["IDENTITY", "EXPERTISE", "BOUNDARY", "PREFERENCE"]
  },
  "gradient": {
    "profile_loaded": true,
    "observe_only": true,
    "pass_rate_7d": 0.94,
    "drift_alert": false,
    "last_evaluated": "2026-02-28T06:44:12Z"
  },
  "memory": {
    "total_episodes": 1847,
    "entities_known": 63,
    "last_ingested": "2026-02-28T06:40:00Z"
  }
}
```

---

## Search Capabilities

### Semantic search (synonyms and paraphrases)
```python
memory_query(query="Quinn was fired")
# Finds episodes containing "Quinn terminated", "Quinn dismissed", etc.
```

### Date-range filtering (new in v0.4)
```python
memory_query(query="trading decisions", after_date="2026-02", before_date="2026-03")
memory_search(query="bot crash", search_type="episodes", after_date="2026-02-15")
```

### Numeric and IP address search (new in v0.4)
```python
memory_search(query="203.0.113.42")        # IP address — CONTAINS fallback auto-triggered
memory_search(query="$50 funding round")    # prices/amounts
memory_search(query="wallet 0xDeadBeef")    # hex/wallet addresses
```

### Fuzzy entity matching (new in v0.4)
```python
memory_search(query="Qinn", search_type="entities")
# Finds "Quinn" via vector similarity fallback when BM25 returns nothing
```

---

## Architecture

OpenStinger runs beside your agent — never inside it. Your agent calls MCP tools. OpenStinger reads session files in the background.

### Why not OpenClaw's native memory — or other memory services?

OpenClaw's built-in memory system and the broader ecosystem of agent memory services each solve part of the problem. OpenStinger uses a different approach for a different set of guarantees. Three concrete differences:

1. **One stack, not many.** Other memory services typically require multiple separate components — a graph store, a separate embedding service, and your own reporting layer stitched together. OpenStinger ships as two containers (FalkorDB + PostgreSQL) with everything included and unified.
2. **SQL-queryable operational database.** Every ingestion job, entity merge, vault classification, and alignment event is logged to PostgreSQL. OpenClaw's native memory and most other services have no equivalent audit trail. Connect any BI tool — Metabase, Grafana, Superset — with zero additional instrumentation.
3. **Alignment evaluation that survives memory compaction.** OpenClaw's `SOUL.md` is a static file injected at session start. If compaction or context pruning drops it mid-session, alignment is gone. OpenStinger's Gradient evaluates every response against the agent's own vault-derived profile — it survives any context operation.

> *Broader ecosystem reference: Mem0, Graphiti, Cognee, Zep — each with different tradeoffs.*

```
OpenClaw    ──┐
Nanobot     ──┤
ZeroClaw    ──┤
NanoClaw    ──┼── MCP / SSE · http://localhost:8766/sse
PicoClaw    ──┤
DeerFlow    ──┤        ▼
Qwen-Agent  ──┤  OpenStinger MCP Server (Python · port 8766)
LangGraph   ──┤
Claude Code ──┤
Cursor      ──┘
               ├── Tier 1  memory_query · memory_add ··········· 11 tools
               ├── Tier 2  vault_promote_now · vault_note_get   11 tools
               └── Tier 3  gradient_alignment_score · ops_status 8 tools
                    │                                     ─────────────────
                    │                                     30 tools total
                    ├── FalkorDB    (graph · vectors · episodic memory)
                    ├── PostgreSQL  (jobs · alignment events · registry)
                    └── vault/      (notes · SHA-256 synced · auditable)
```

Session files are read-only. OpenStinger never writes to your agent's files.

---

## Session Formats

**OpenClaw format** (`session_format: openclaw`):
OpenStinger parses OpenClaw v3 JSONL files, extracting user messages and assistant responses. Thinking blocks, tool calls, and metadata are skipped.

**DeerFlow format** (`session_format: deerflow`):
OpenStinger parses DeerFlow 2.0 `thread.json` conversation files via the `DeerFlowSessionReader` adapter (v0.7+). Configure `sessions_dir` to point at your DeerFlow project's `threads/` directory.

**Qwen-Agent format** (`session_format: qwen_agent`):
OpenStinger parses Qwen-Agent JSONL session files written by `QwenSessionWriter` (v0.7+). Each file is a per-session conversation log. Configure `sessions_dir` to match the `session_dir` path in your `OpenStingerQwenAgent` config.

**Simple format** (`session_format: simple`):
One JSON object per line — works with any runtime that can write JSONL:
```json
{"content": "conversation text here", "source": "conversation", "valid_at": 1234567890}
```

---

## Browser UIs

| URL | Tool | Start command |
|---|---|---|
| `http://localhost:3000` | FalkorDB Browser (visual graph) | `docker compose up -d` (auto-starts) |
| `http://localhost:8080` | Adminer (PostgreSQL inspector) | `docker compose up -d` (auto-starts) |

**FalkorDB Browser login:** host `host.docker.internal` · port `6379` · password: whatever you set in `.env` (blank = no password)

**Adminer login:** System: `PostgreSQL` · Server: `host.docker.internal` · Username/Password/Database: use values from your `.env` and `config.yaml`

> Use `host.docker.internal` NOT `localhost` — both UIs run inside Docker.

---

## Upgrading Tiers

```
Tier 1  python -m openstinger.mcp.server                  ← start here
   ↓    (~1,000+ episodes ingested — usually 1–2 days)
Tier 2  python -m openstinger.scaffold.mcp.server          ← vault activates
   ↓    (vault builds identity + constraint notes — ~1–2 weeks)
Tier 3  python -m openstinger.gradient.mcp.server          ← alignment activates (observe-only first)
```

Each tier includes all lower tiers. Running Tier 3 gives you all 30 tools.

---

## Configuration Reference

```yaml
agent_name: main               # your agent's name
agent_namespace: main          # namespace for graph isolation

falkordb:
  host: localhost
  port: 6379
  password: ""                 # leave blank for local dev

operational_db:
  provider: postgresql            # postgresql (default, v0.6+) | sqlite
  postgresql_url: "postgresql+asyncpg://your_user:your_password@localhost:5432/your_db"
  # sqlite_path: ".openstinger/openstinger.db"  # uncomment to use SQLite instead

llm:
  provider: anthropic          # anthropic | openai (for OpenAI-compatible)
  model: claude-sonnet-4-6
  fast_model: claude-haiku-4-5-20251001
  embedding_model: text-embedding-3-small
  # For Novita / DeepSeek / other OpenAI-compatible:
  # provider: openai
  # llm_base_url: "https://api.novita.ai/v3/openai"
  # embedding_base_url: "https://api.novita.ai/v3/openai"

ingestion:
  sessions_dir: "/path/to/sessions"  # REQUIRED: path to your agent's JSONL session files
  session_format: openclaw            # openclaw | simple
  poll_interval_seconds: 10
  concurrency: 5                      # parallel episode processing (v0.5+)

vault:                              # Tier 2
  classification_interval_seconds: 300

gradient:                           # Tier 3
  enabled: false
  observe_only: true                # always start in observe mode

mcp:
  transport: sse                    # sse (recommended) | stdio
  tcp_port: 8766                    # default production port (8765 may be reserved on Windows)
```

---

## Troubleshooting

**Server exits immediately after starting**
Check the log (`.openstinger/openstinger.log`). Common causes:
- FalkorDB not reachable: verify `docker ps | grep falkordb` and test with `docker exec openstinger_falkordb redis-cli ping`
- Port in use: change `mcp.tcp_port` in `config.yaml` to a free port

**FalkorDB password issue**
If you set `FALKORDB_PASSWORD` in `.env` and FalkorDB starts without auth anyway, the password contains `#`. Rename the password to avoid `#` characters and recreate the container:
```bash
docker compose down
docker volume rm openstinger_falkordb_data
docker compose up -d
```

**Port already in use (WinError 10048 / 10013)**
Windows may have the port reserved (Hyper-V, elevated process). Simply use a different port:
```yaml
mcp:
  tcp_port: 8766   # or 8767, 8768 — whatever is free
```
Update your MCP client config to match.

**Semantic search not working ("fired" doesn't find "terminated")**
`memory_query` (not `memory_search`) runs vector search. Use `memory_query` for semantic recall. All episodes ingested with v0.4+ have vector embeddings.

**"February 2026" returns nothing**
Use the `after_date` / `before_date` parameters in `memory_query`:
```python
memory_query(query="decisions", after_date="2026-02", before_date="2026-03")
```
Episodes from v0.4+ also have `valid_at_human` BM25-indexed for date keyword search.

---

## Testing

```bash
# Without FalkorDB (fast)
pytest tests/ -m "not integration"

# Full suite (FalkorDB must be running)
pytest tests/

# Specific tier
pytest tests/ -m tier1
pytest tests/ -m tier2
pytest tests/ -m tier3
```

36+ tests passing across all tiers.

## Production & Enterprise

> **OpenStinger is the only \*Claw-compatible memory system with a queryable operational database.**
>
> Every episode, entity, classification decision, and alignment event is logged to PostgreSQL — ready for dashboards, audits, and compliance reports. Not exported on request. Always on. Always queryable.

Connect any BI tool — Metabase, Grafana, Superset, or `psql` — and get full operational visibility with zero additional instrumentation. OpenStinger logs everything to a 12-table schema from the moment it starts.

### Key queries

```sql
-- Who has this agent's memory touched? (PII / people audit)
SELECT name, entity_type, episode_count
FROM entity_registry
WHERE entity_type = 'PERSON'
ORDER BY episode_count DESC;

-- Alignment verdict distribution this week
SELECT verdict, COUNT(*) AS count
FROM alignment_events
WHERE evaluated_at > EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days')
GROUP BY verdict;

-- Knowledge confidence by category (vault health)
SELECT category,
       COUNT(*) FILTER (WHERE stale = false) AS active_notes,
       ROUND(AVG(confidence)::numeric, 2) AS avg_confidence
FROM vault_notes GROUP BY category;

-- Episode volume per day (ingestion health)
SELECT DATE(TO_TIMESTAMP(created_at)) AS day, COUNT(*) AS episodes
FROM episode_log GROUP BY day ORDER BY day DESC LIMIT 7;
```

> **Adminer** (bundled in `docker compose`) gives a point-and-click interface to every table at `http://localhost:8080`.

---

## Memory Portability


> **Your agent changes. The memory doesn't have to.**

OpenStinger stores everything in two Docker volumes:

- `falkordb_data` — the temporal knowledge graph: all entities, relationships, episodes
- `postgres_data` — the operational database: vault notes, alignment events, registry

Both are standard Docker volumes. Export, move, import — full memory transfers to a new host, a new runtime, or a new cloud provider. The agent framework is irrelevant because memory is decoupled from it.

### Backup

```bash
# FalkorDB snapshot (atomic — run first)
docker exec openstinger_falkordb redis-cli BGSAVE

# Export both volumes
docker run --rm -v openstinger_falkordb_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/falkordb.tar.gz /data

docker run --rm -v openstinger_postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres.tar.gz /data
```

### Restore on any host

```bash
docker run --rm -v openstinger_falkordb_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/falkordb.tar.gz -C /

docker run --rm -v openstinger_postgres_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres.tar.gz -C /

docker compose up -d   # starts with full memory intact
```

### What this enables

| Scenario | Result |
|---|---|
| **Agent migration** | Move from Mac Mini to cloud VM — full memory, vault, alignment history preserved |
| **Runtime swap** | Switch from OpenClaw to NanoClaw — OpenStinger memory transfers completely |
| **Agent cloning** | Snapshot production into dev — test against real accumulated knowledge |
| **Multi-agent bootstrap** | Clone a senior agent's vault into a new specialist — starts with wisdom, not zero |
| **Emergency recovery** | Host fails — restore volumes on any new machine, agent resumes |
| **Memory rollback** | Snapshot before risky vault changes — restore if classification results degrade |

No other *Claw memory system separates memory from the runtime cleanly enough to make this possible.

---

## Cloud Deployment

OpenStinger runs anywhere Python and Docker run. Deploy on a cloud VM and all your agents — on-prem, edge devices, other cloud instances — connect via MCP SSE.

```
[Mac Mini agent]         ──HTTP SSE──►
[EC2 agent]              ──HTTP SSE──►  OpenStinger (cloud VM · port 8766)
[PicoClaw on $10 device] ──HTTP SSE──►
[NanoClaw swarm agent]   ──HTTP SSE──►
```

For remote agents that generate session files locally, use the `memory_add` MCP tool to write episodes directly — no local file path required.

---

## License

MIT — see [LICENSE](LICENSE)
