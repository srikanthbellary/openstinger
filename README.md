# OpenStinger

<p align="center">
  <img src="assets/OpenStinger_Logo_v3_transparent.png" alt="OpenStinger" width="480">
</p>

<p align="center">
  <strong>Persistent memory, self-knowledge, and alignment for autonomous AI agents.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python 3.10+"></a>
  <a href="https://falkordb.com"><img src="https://img.shields.io/badge/FalkorDB-1.6%2B-orange?style=for-the-badge" alt="FalkorDB"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-1.26%2B-green?style=for-the-badge" alt="MCP"></a>
</p>

**OpenStinger** gives autonomous AI agents persistent memory, structured self-knowledge, and alignment evaluation — all exposed as MCP tools the agent calls natively.

Built on [FalkorDB](https://falkordb.com) (graph + vector database) and [Model Context Protocol](https://modelcontextprotocol.io). No SDK changes to your agent. No vendor lock-in. Works with any MCP-compatible runtime (OpenClaw, Claude Code, etc.).

---

## What It Does

Three additive tiers. Start with Tier 1 and unlock the rest as data accumulates.

| Tier | Name | Tools | What it gives your agent |
|---|---|---|---|
| **Tier 1** | Memory Harness | 9 | Bi-temporal episodic memory. Every session ingested automatically. Hybrid BM25 + vector semantic search. Date filtering. Numeric/IP search. |
| **Tier 2** | StingerVault | 19 | Autonomous distillation of sessions into structured self-knowledge: identity, domain, methodology, preferences, constraints. External document ingestion (URL, PDF, YouTube). |
| **Tier 3** | Gradient | 27 | Synchronous alignment evaluation before every response. Drift detection. Correction engine. 3 new observability tools (v0.6). Starts in observe-only mode. |

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

**Tier 1 only** (memory, 9 tools):
```bash
python -m openstinger.mcp.server
```

**All tiers** (memory + vault + alignment, 27 tools):
```bash
python -m openstinger.gradient.mcp.server
```

### 5. Connect your agent

In your MCP client config (e.g. `mcporter.json`):
```json
{
  "mcpServers": {
    "openstinger": {
      "baseUrl": "http://localhost:8766/sse"
    }
  }
}
```

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

## MCP Tools

### Tier 1 — Memory (9 tools)

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

### Tier 2 — StingerVault (+10 tools, 19 total)

`vault_status` · `vault_sync_now` · `vault_stats` · `vault_promote_now` · `vault_note_list` · `vault_note_get` · `knowledge_ingest` · `namespace_create` · `namespace_list` · `namespace_archive`

### Tier 3 — Gradient (+8 tools, 27 total)

| Tool | What it does |
|---|---|
| `gradient_status` | Gradient health, profile state, observe_only flag |
| `gradient_alignment_score` | Evaluate a response — returns score + verdict |
| `gradient_drift_status` | Rolling window alignment stats |
| `gradient_alignment_log` | Recent alignment evaluation log |
| `gradient_alert` | Current drift alert status |
| `ops_status` ⭐ | Single-call dashboard: vault notes + gradient pass rate + drift state |
| `gradient_history` ⭐ | Last N alignment verdicts with scores from PostgreSQL |
| `drift_status` ⭐ | Behavioral drift window history from PostgreSQL |

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
memory_search(query="192.168.1.100")        # IP address — CONTAINS fallback auto-triggered
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

```
Your Agent (any MCP-compatible runtime)
    │
    │  Model Context Protocol · SSE · http://localhost:8766/sse
    ▼
OpenStinger MCP Server (Python process on your machine)
    ├── Tier 1  memory_query · memory_add · memory_search ··········  9 tools
    ├── Tier 2  vault_promote_now · knowledge_ingest · namespace_*  +10 tools
    └── Tier 3  gradient_alignment_score · ops_status · drift_status  +8 tools
         │                                                      ──────────────
         │                                                      27 tools total
         ├── FalkorDB    (graph DB · episodic memory · knowledge vault · vectors)
         ├── PostgreSQL  (ingestion jobs · alignment events · agent registry)
         └── vault/      (markdown notes · human-editable · SHA-256 synced)
```

Session files are read-only. OpenStinger never writes to your agent's files.

---

## Session Formats

**OpenClaw format** (`session_format: openclaw`):
OpenStinger parses OpenClaw v3 JSONL files, extracting user messages and assistant responses. Thinking blocks, tool calls, and metadata are skipped.

**Simple format** (`session_format: simple`):
One JSON object per line:
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

Each tier includes all lower tiers. Running Tier 3 gives you all 27 tools.

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

---

## License

MIT — see [LICENSE](LICENSE)
