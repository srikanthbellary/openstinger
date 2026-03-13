# DeerFlow 2.0 — OpenStinger Integration Guide

**OpenStinger version:** v0.7+  
**DeerFlow version:** 2.0  
**Integration type:** MCP tool access (immediate) + automatic session ingestion (v0.7)  
**Effort to first working integration:** ~15 minutes

---

## What This Integration Does

DeerFlow 2.0 ships with a `memory.json` file for cross-session persistence. It is a flat JSON key-value store: no temporal history, no entity deduplication, no conflict resolution, no semantic search. It works for simple preference storage. It does not scale as sessions accumulate.

OpenStinger replaces `memory.json` as DeerFlow's memory backend. When connected, DeerFlow's agent:

- Calls `memory_add()` to store facts — bi-temporal, deduplicated, conflict-resolved
- Calls `memory_query()` to retrieve context — hybrid BM25 + vector semantic search
- Gets structured self-knowledge via `vault_note_list()` (Tier 2)
- Gets alignment evaluation via `gradient_alignment_score()` (Tier 3)
- Gets a single-call operational dashboard via `ops_status()`

OpenStinger also ingests DeerFlow's `thread.json` session files automatically — every conversation turn is read into the temporal graph on OpenStinger's polling schedule, building the agent's memory without any explicit `memory_add()` calls.

---

## Prerequisites

- **DeerFlow 2.0** installed and running (`make docker-start` or `make dev`)
- **OpenStinger v0.7** installed (`pip install -e "."` from the openstinger repo)
- **Docker Desktop** running (for FalkorDB + PostgreSQL)
- **Python 3.10+**
- One API key: Anthropic, OpenAI, or any OpenAI-compatible provider (Novita, DeepSeek, etc.)

---

## Step 1 — Start OpenStinger

```bash
# From your openstinger directory
docker compose up -d           # starts FalkorDB + PostgreSQL + Adminer
python -m openstinger.gradient.mcp.server   # starts all 28 tools on port 8766
```

Verify it's running:

```bash
curl http://localhost:8766/sse
# Should return: data: {"type": "endpoint", ...}
```

---

## Step 2 — Connect OpenStinger MCP to DeerFlow

DeerFlow reads MCP server config from `extensions_config.json` in the project root.

If `extensions_config.json` does not exist, create it from the example:

```bash
cp extensions_config.example.json extensions_config.json
```

Add the OpenStinger MCP server to the `mcpServers` block:

```json
{
  "mcpServers": {
    "openstinger": {
      "type": "url",
      "url": "http://host.docker.internal:8766/sse",
      "name": "openstinger-mcp"
    }
  }
}
```

> **Why `host.docker.internal` and not `localhost`?**  
> DeerFlow's agent runs inside a Docker container. From inside Docker, `localhost` refers to the container itself. `host.docker.internal` is Docker's built-in hostname that resolves to the host machine where OpenStinger is running.  
> If you are running DeerFlow in **local development mode** (not Docker), use `http://localhost:8766/sse` instead.

Restart DeerFlow for the config to take effect:

```bash
make docker-start   # or Ctrl+C and make dev
```

At this point all 27 OpenStinger MCP tools are callable from DeerFlow's agent. Automatic session ingestion is configured in Step 3.

---

## Step 3 — Configure OpenStinger to Ingest DeerFlow Sessions

OpenStinger ingests DeerFlow's `thread.json` session files automatically. Tell it where to find them.

Edit your OpenStinger `config.yaml`:

```yaml
agent_name: deerflow-agent       # name this anything meaningful
agent_namespace: deerflow-agent  # used for graph namespace isolation

ingestion:
  sessions_dir: "/absolute/path/to/your/deer-flow/.deer-flow/threads"
  session_format: deerflow        # tells OpenStinger to use DeerFlowSessionReader
  poll_interval_seconds: 10       # check for new thread messages every 10 seconds
  concurrency: 5

falkordb:
  host: localhost
  port: 6379
  password: ""

operational_db:
  provider: postgresql
  postgresql_url: "postgresql+asyncpg://openstinger:your_postgres_password@localhost:5432/openstinger"

llm:
  provider: anthropic
  model: claude-sonnet-4-6
  fast_model: claude-haiku-4-5-20251001
  embedding_model: text-embedding-3-small
```

> **Finding your DeerFlow threads directory:**  
> By default DeerFlow stores threads at `.deer-flow/threads/` relative to where you cloned the repo.  
> Each thread gets its own subdirectory: `.deer-flow/threads/{thread_id}/thread.json`  
> Use the absolute path — do not use `~` (tilde does not expand in all contexts).

Restart the OpenStinger MCP server after changing `config.yaml`:

```bash
python -m openstinger.gradient.mcp.server
```

Check ingestion is running:

```bash
python -m openstinger.cli progress
```

---

## Step 4 — Drop In the OpenStinger Skill File

The skill file teaches DeerFlow's agent when and how to use OpenStinger tools. Without it, the agent has the tools available but no guidance on when to call them.

Copy the OpenStinger skill file into DeerFlow's skills directory **inside the sandbox container**:

```bash
# From your DeerFlow project root:
mkdir -p skills/public/openstinger
cp /path/to/openstinger/integrations/AGENT_SKILL_TEMPLATE.md skills/public/openstinger/SKILL.md
# (the skill file lives at openstinger/integrations/AGENT_SKILL_TEMPLATE.md in the OpenStinger repo)
```

DeerFlow loads skills from `/mnt/skills/` inside the container, which maps to `skills/` in your project root. The skill is loaded progressively — only when the agent's task needs memory or alignment tools.

---

## Step 5 — Replace memory.json (Optional but Recommended)

DeerFlow's built-in `memory.json` and OpenStinger can run **alongside each other** at first. When you're confident in OpenStinger, disable `memory.json` to prevent the agent from writing to both.

**To disable DeerFlow's native memory persistence**, add to your DeerFlow `config.yaml`:

```yaml
# config.yaml (DeerFlow's config, not OpenStinger's)
memory:
  enabled: false
```

> **⚠ Verify the exact config key** against your DeerFlow 2.0 version before disabling. The memory config key may differ — check DeerFlow's `backend/docs/CONFIGURATION.md`.

Once disabled, all memory operations go through OpenStinger's MCP tools. The agent will call `memory_add()` and `memory_query()` as guided by the skill file.

---

## Three Adoption Modes

OpenStinger and DeerFlow's native memory can coexist or replace each other. Choose the mode that fits where you are.

### Mode 1 — Alongside (Start Here)

Both OpenStinger MCP tools and DeerFlow's `memory.json` are active. The agent has access to all OpenStinger tools but DeerFlow's framework may still write to `memory.json` for its own preferences.

**When:** Day 1. Add OpenStinger, change nothing else.  
**Config:** Complete Steps 1–4. Do not disable DeerFlow memory.  
**Benefit:** Zero risk. Evaluate OpenStinger's memory quality before committing.

### Mode 2 — Primary

OpenStinger is the primary memory backend. The agent's skill file instructs it to call `memory_query()` first on every task. DeerFlow's `memory.json` is still present as a fallback.

**When:** After ~500 episodes are ingested — usually 3–5 days of active use.  
**Config:** Update the skill file: add "Always call `memory_query()` before starting any research task."  
**Benefit:** Full OpenStinger retrieval quality with a safety net.

### Mode 3 — Exclusive

DeerFlow's native `memory.json` is disabled. OpenStinger is the only memory backend.

**When:** After Tier 2 vault has built up structured knowledge (identity + domain notes present).  
**Config:** Set `memory.enabled: false` in DeerFlow's config.yaml.  
**Benefit:** Clean, single source of truth. No redundant writes.

---

## Verifying the Integration

### Check MCP tools are available in DeerFlow

In a DeerFlow chat session, ask the agent:

> "List the MCP tools you have available from the openstinger server."

The agent should list the OpenStinger tools (memory_query, memory_add, vault_status, ops_status, etc.).

### Check ingestion is running

```bash
# From your OpenStinger directory:
python -m openstinger.cli progress
```

Expected output after a few DeerFlow sessions:

```
Ingestion:  12 / 12 files processed  [████████████] 100%
Episodes:   847 in FalkorDB
Entities:   63 known
Vault:      identity: 4  domain: 11  methodology: 3  preference: 7  constraint: 2
Gradient:   observe_only=true  pass_rate_7d=0.97  drift_alert=false
```

### Check sessions are being read

```bash
# In Adminer at http://localhost:8080
SELECT file_path, entries_read, last_read_at
FROM ingestion_cursor
WHERE agent_id = 'deerflow-agent'
ORDER BY last_read_at DESC;
```

Each DeerFlow thread should appear as a row with a non-zero `entries_read`.

### Run a semantic memory query

From the DeerFlow chat interface, ask the agent something that references a past session:

> "What research tasks have we worked on recently?"

The agent should call `memory_query()` and return results from OpenStinger's temporal graph — not from `memory.json`.

---

## Browser UIs

| URL | Tool | What it shows |
|---|---|---|
| `http://localhost:3000` | FalkorDB Browser | Visual graph: episodes, entities, relationships |
| `http://localhost:8080` | Adminer | PostgreSQL: all tables, ingestion jobs, alignment events |

**FalkorDB Browser login:** host `host.docker.internal` · port `6379` · leave password blank (unless you set one in `.env`)

**Adminer login:** System: PostgreSQL · Server: `host.docker.internal` · Username: `openstinger` · Password: `your_postgres_password` (value from your `.env`) · Database: `openstinger`

---

## How Session Ingestion Works

DeerFlow writes session state to `.deer-flow/threads/{thread_id}/thread.json`. This is a JSON object containing a `messages` array with all conversation turns.

OpenStinger's `DeerFlowSessionReader` (v0.7):

1. Scans the `threads/` directory for `thread.json` files every `poll_interval_seconds`
2. For each file, reads the `messages` array
3. Filters to `user` and `assistant` roles only (skips tool_use, tool_result, system)
4. Tracks position by message ID in the `ingestion_cursor` table (not byte-offset, because `thread.json` is rewritten atomically on each update rather than appended)
5. Feeds new messages into the temporal engine as episodes
6. Entity deduplication, conflict resolution, and embedding happen on the normal ingestion pipeline — identical to OpenClaw sessions

**What gets ingested from each message:**
- `content` — the text of the message
- `role` — user or assistant
- `created_at` — used as `valid_at` (when the event actually happened) for bi-temporal correctness
- `thread_id` — used as `session_id` for namespace isolation

---

## Using OpenAI-Compatible Providers

OpenStinger works with any OpenAI-compatible API. If you are already running DeerFlow with a local model (Ollama, vLLM) or a provider like Novita or DeepSeek, you can use the same endpoint for OpenStinger:

```yaml
# config.yaml (OpenStinger)
llm:
  provider: openai
  model: deepseek/deepseek-v3.2
  llm_base_url: "https://api.novita.ai/v3/openai"
  embedding_model: qwen/qwen3-embedding-8b
  embedding_provider: openai
  embedding_base_url: "https://api.novita.ai/v3/openai"
```

Set `OPENAI_API_KEY` in `.env` to your provider's API key.

---

## Troubleshooting

### "openstinger MCP tools not available in DeerFlow"

- Confirm OpenStinger MCP server is running: `curl http://localhost:8766/sse`
- Confirm `extensions_config.json` has the correct URL
- Restart DeerFlow after editing `extensions_config.json`
- If DeerFlow runs in Docker, ensure you used `host.docker.internal` not `localhost`

### "No episodes appearing in FalkorDB after DeerFlow sessions"

- Confirm `sessions_dir` in OpenStinger `config.yaml` is the correct absolute path to `.deer-flow/threads/`
- Confirm at least one thread has a `thread.json` file in it
- Check OpenStinger logs: `.openstinger/openstinger.log`
- Run `python -m openstinger.cli progress` — if ingestion shows 0 files, the path is wrong

### "DeerFlow agent not calling memory_query"

- Confirm the SKILL.md is in `skills/public/openstinger/SKILL.md` in your DeerFlow project root
- The skill is loaded progressively — it only activates when the task needs memory tools
- Try explicitly asking the agent: "Search your memory for anything related to [topic]"

### "thread.json read errors in logs"

- DeerFlow writes `thread.json` atomically but there may be a brief window during write
- OpenStinger's reader includes a partial-read guard — if the file fails JSON parsing, it skips it and retries next cycle
- These errors are harmless and self-correcting

### "MCP connection refused when DeerFlow runs in Docker"

- Change URL to `http://host.docker.internal:8766/sse` in `extensions_config.json`
- On Linux hosts, `host.docker.internal` may not be set automatically — add it to `docker-compose.yml`:
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```

---

## What OpenStinger Does That memory.json Cannot

| Capability | memory.json | OpenStinger |
|---|---|---|
| Cross-session persistence | ✅ flat key/value | ✅ bi-temporal graph |
| Semantic search | ❌ no retrieval | ✅ BM25 + vector hybrid |
| Temporal correctness | ❌ last-write-wins | ✅ valid_at + recorded_at separate |
| Entity deduplication | ❌ duplicates accumulate | ✅ 3-stage LSH + LLM confirm |
| Conflict resolution | ❌ silent overwrite | ✅ LLM-based with temporal precedence |
| Structured self-knowledge | ❌ raw key/value | ✅ identity / domain / constraint vault notes |
| Alignment evaluation | ❌ none | ✅ 4-dimensional per-response scoring |
| Audit trail | ❌ none | ✅ 12-table PostgreSQL audit log |
| Memory portability | ❌ one file | ✅ Docker volumes — move to any host |
| Scales with session count | ❌ context injection grows unbounded | ✅ semantic retrieval — only relevant episodes returned |

---

## Reference — All 27 OpenStinger Tools

### Tier 1 — Memory (9 tools)

| Tool | When to call |
|---|---|
| `memory_add(content)` | Store an important fact explicitly |
| `memory_query(query, limit=5)` | Semantic recall — "what were we working on last week?" |
| `memory_search(query)` | Exact terms, IP addresses, numbers, wallet addresses |
| `memory_get_entity(entity_id)` | Fetch a specific entity and its relationships |
| `memory_get_episode(episode_id)` | Fetch a specific episode by ID |
| `memory_job_status()` | Check ingestion job queue state |
| `memory_ingest_now()` | Trigger immediate ingestion cycle |
| `memory_namespace_status(agent_namespace)` | Episode/entity/edge counts for this namespace |
| `memory_list_agents()` | List all registered agent namespaces |

### Tier 2 — StingerVault (11 additional tools, 20 total)

| Tool | When to call |
|---|---|
| `vault_status()` | Check vault health and note counts |
| `vault_sync_now()` | Trigger immediate classification cycle |
| `vault_stats()` | Counts per note category |
| `vault_promote_now(content, category)` | Manually promote a fact to structured knowledge |
| `vault_note_list(category)` | Browse notes: identity / domain / methodology / preference / constraint |
| `vault_note_get(note_id)` | Read a specific vault note |
| `vault_note_add(content, category)` | **v0.7** — manually seed a vault note (identity, constraint, etc.) |
| `knowledge_ingest(source, source_type)` | Ingest URL / PDF / YouTube / text into knowledge graph |
| `namespace_create(agent_id)` | Create a new agent namespace |
| `namespace_list()` | List all namespaces |
| `namespace_archive(agent_id)` | Archive a namespace |

### Tier 3 — Gradient (8 additional tools, 28 total)

| Tool | When to call |
|---|---|
| `ops_status()` | **Start of every session** — vault notes + gradient state + episode count |
| `gradient_status()` | Gradient health and observe_only flag |
| `gradient_alignment_score(content)` | Evaluate a response before delivery |
| `gradient_drift_status()` | Rolling window alignment trend |
| `gradient_alignment_log(since_timestamp)` | Recent alignment evaluation events |
| `gradient_alert()` | Current drift alert status |
| `gradient_history(n=10)` | Last N alignment verdicts with scores |
| `drift_status()` | Behavioral drift history |

---

*OpenStinger v0.7 · DeerFlow 2.0 Integration Guide*  
*MIT License · https://github.com/srikanthbellary/openstinger*
