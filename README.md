# OpenStinger

<p align="center">
  <img src="assets/OpenStinger_Logo_v3_transparent.png" alt="OpenStinger" width="480">
</p>

<p align="center">
  <strong>Persistent memory, self-knowledge, and alignment for autonomous AI agents.</strong>
</p>

<p align="center">
  <a href="https://github.com/srikanthbellary/openstinger/actions"><img src="https://img.shields.io/github/actions/workflow/status/srikanthbellary/openstinger/ci.yml?branch=main&style=for-the-badge" alt="CI"></a>
  <a href="https://github.com/srikanthbellary/openstinger/releases"><img src="https://img.shields.io/github/v/release/srikanthbellary/openstinger?include_prereleases&style=for-the-badge" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python 3.10+"></a>
</p>

**OpenStinger** is an open-source memory and alignment harness for autonomous AI agents. It gives your agents persistent episodic memory that grows over time, a self-building knowledge vault distilled from their own sessions, and an alignment layer that keeps behavior consistent with stated values — all exposed as MCP tools your agent can call natively.

Built on [FalkorDB](https://falkordb.com) (graph database), [Model Context Protocol](https://modelcontextprotocol.io), and an OpenAI-compatible API layer. Works alongside any agent runtime that supports MCP.

---

## What It Does

OpenStinger installs additively across three tiers. Start with Tier 1 and upgrade when ready.

| Tier | Name | What it adds |
|---|---|---|
| **Tier 1** | Memory Harness | Bi-temporal episodic memory on FalkorDB. Conflict-resolving knowledge graph. **9 MCP tools.** |
| **Tier 2** | StingerVault | Autonomous classification of agent sessions into structured self-knowledge. **15 MCP tools total.** |
| **Tier 3** | Gradient | Synchronous alignment evaluation before every response. Drift detection. Correction engine. **20 MCP tools total.** |

---

## Quick Start

**Requirements:** Python 3.10+, Docker Desktop

```bash
# 1. Clone and set up
git clone https://github.com/srikanthbellary/openstinger.git
cd openstinger
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env          # add your API keys
cp config.yaml.example config.yaml   # set sessions_dir to your agent's session folder

# 3. Start FalkorDB
docker compose up -d

# 4. Start the MCP server (Tier 1)
python -m openstinger.mcp.server
```

Or use the startup script:

```bash
# Windows
scripts\start.bat

# Linux / macOS / Git Bash
./scripts/start.sh
```

---

## MCP Tools

### Tier 1 — Memory (9 tools)

| Tool | Description |
|---|---|
| `memory_add` | Store an episode manually |
| `memory_query` | Hybrid BM25 + vector semantic search |
| `memory_search` | BM25 keyword search across episodes, entities, or facts |
| `memory_get_entity` | Fetch an entity and its current relationships |
| `memory_get_episode` | Fetch a specific episode by UUID |
| `memory_job_status` | Check ingestion job status |
| `memory_ingest_now` | Trigger immediate session ingestion |
| `memory_namespace_status` | Health stats: episode / entity / edge counts |
| `memory_list_agents` | List all registered agent namespaces |

### Tier 2 — StingerVault (+6 tools, 15 total)

`vault_status` · `vault_sync_now` · `vault_stats` · `vault_promote_now` · `vault_note_list` · `vault_note_get`

### Tier 3 — Gradient (+5 tools, 20 total)

`gradient_status` · `gradient_alignment_score` · `gradient_drift_status` · `gradient_alignment_log` · `gradient_alert`

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.10 or 3.11 recommended |
| Docker Desktop | Runs FalkorDB and the browser UI |
| LLM API key | Anthropic or any OpenAI-compatible provider (Novita, DeepSeek, etc.) |
| Embedding API key | OpenAI or any OpenAI-compatible embedding endpoint |

### Install

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[dev,tools]"    # + Datasette browser for SQLite inspection
```

### Configure

Edit `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-...      # or use any OpenAI-compatible provider
OPENAI_API_KEY=sk-...             # for embeddings
FALKORDB_PASSWORD=your-password
```

Edit `config.yaml` — key settings:
```yaml
agent_name: main
agent_namespace: main

llm:
  provider: openai
  model: deepseek/deepseek-v3.2
  llm_base_url: "https://api.novita.ai/v3/openai"   # optional: use any OpenAI-compatible API
  embedding_model: qwen/qwen3-embedding-8b
  embedding_base_url: "https://api.novita.ai/v3/openai"

ingestion:
  sessions_dir: "/path/to/your/agent/sessions"
  session_format: openclaw   # or: simple

mcp:
  transport: sse
  tcp_port: 8765
```

---

## Upgrading Tiers

```
Tier 1  python -m openstinger.mcp.server              ← start here
   ↓    (after ~1,000 episodes ingested)
Tier 2  python -m openstinger.scaffold.mcp.server     ← StingerVault activates
   ↓    (after vault builds identity notes, ~1–2 weeks)
Tier 3  python -m openstinger.gradient.mcp.server     ← starts in observe_only mode
```

---

## Browser UIs

Two browser tools for inspecting memory state:

| URL | Tool | What it shows |
|---|---|---|
| `http://localhost:3000` | FalkorDB Browser | Visual knowledge graph — entities, facts, relationships |
| `http://localhost:8001` | Datasette | SQLite operational DB — all tables |

FalkorDB Browser starts automatically with `docker compose up -d`.
Login: host `host.docker.internal` · port `6379` · username `default` · password from `.env`.

---

## Session Formats

OpenStinger parses agent session JSONL files automatically.

**OpenClaw format** (`session_format: openclaw`):
Reads user messages and assistant responses; skips thinking blocks, tool calls, and metadata.

**Simple format** (`session_format: simple`):
```json
{"content": "...", "source": "conversation", "valid_at": 1234567890}
```

---

## Architecture

```
Your machine
├── OpenStinger MCP server (Python, SSE on port 8765)
│   ├── reads: agent session files (read-only)
│   └── writes: FalkorDB graph + SQLite operational DB
│
Docker
├── FalkorDB (port 6379)         ← graph database
└── FalkorDB Browser (port 3000) ← visual UI
```

---

## Multi-Agent Setup

Each agent gets its own isolated namespace. Run separate server instances with different configs:

```yaml
# config-dev.yaml
agent_name: dev
agent_namespace: dev
ingestion:
  sessions_dir: "/path/to/dev/sessions"
```

---

## Testing

```bash
# All tests (FalkorDB must be running)
pytest tests/

# Without FalkorDB
pytest tests/ -m "not integration"
```

80 tests across Tier 1, 2, and 3.

---

## License

MIT — see [LICENSE](LICENSE)
