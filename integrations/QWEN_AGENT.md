# Qwen-Agent — OpenStinger Integration Guide

**OpenStinger version:** v0.7+  
**Qwen-Agent version:** 0.0.10+  
**Integration type:** MCP tool access (immediate) + automatic session ingestion via QwenSessionWriter (v0.7)  
**Effort to first working integration:** ~20 minutes

---

## What This Integration Does

Qwen-Agent is stateless by design. Every call to `agent.run(messages)` passes the full conversation history as a parameter. Nothing is written to disk between calls. Each new session starts with no memory of previous ones.

OpenStinger gives Qwen-Agent persistent, temporally-correct cross-session memory. When integrated:

- Every conversation turn is captured by `QwenSessionWriter` and written to JSONL
- OpenStinger's ingestion scheduler reads those files and builds the temporal knowledge graph
- The agent calls `memory_query()` to retrieve semantically relevant context from any past session
- Structured self-knowledge builds automatically in the vault (Tier 2)
- Alignment evaluation is available per-response (Tier 3)

Unlike DeerFlow — which already writes `thread.json` files OpenStinger can read directly — Qwen-Agent writes nothing to disk by default. The `QwenSessionWriter` and `OpenStingerQwenAgent` wrapper in v0.7 bridge that gap.

---

## Prerequisites

- **Qwen-Agent 0.0.10+** installed: `pip install qwen-agent>=0.0.10`
- **OpenStinger v0.7** installed: `pip install -e ".[qwen-agent]"` (from the openstinger repo)
- **Docker Desktop** running (for FalkorDB + PostgreSQL)
- **Python 3.10+**
- One API key for OpenStinger's LLM: Anthropic, OpenAI, or any OpenAI-compatible provider
- One API key (or local endpoint) for Qwen-Agent's LLM: Qwen API, Novita, local vLLM/Ollama

> **Installing together:**
> ```bash
> pip install "openstinger[qwen-agent]"
> ```
> This installs OpenStinger v0.7 and pulls in `qwen-agent>=0.0.10` as a dependency.

---

## Step 1 — Start OpenStinger

```bash
# From your openstinger directory
docker compose up -d                            # FalkorDB + PostgreSQL + Adminer
python -m openstinger.gradient.mcp.server       # all 30 tools on port 8766
```

Verify:

```bash
curl http://localhost:8766/sse
# Expected: data: {"type": "endpoint", ...}
```

---

## Step 2 — Configure OpenStinger for Qwen-Agent Sessions

Edit your OpenStinger `config.yaml`. The key change is `session_format: qwen_agent` and pointing `sessions_dir` to wherever `QwenSessionWriter` will write its JSONL files.

```yaml
agent_name: qwen-agent           # name this anything meaningful
agent_namespace: qwen-agent      # used for graph namespace isolation

ingestion:
  sessions_dir: "/absolute/path/to/qwen_sessions"   # QwenSessionWriter writes here
  session_format: qwen_agent                         # routes to QwenSessionReader
  poll_interval_seconds: 10
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

> **The `sessions_dir` path must match the `sessions_dir` you pass to `OpenStingerQwenAgent` in Step 3.**  
> Both need to point to the same directory. OpenStinger reads from it; the writer writes to it.  
> Use an absolute path. Do not use `~` (tilde).

Restart the OpenStinger MCP server after any `config.yaml` change:

```bash
python -m openstinger.gradient.mcp.server
```

---

## Step 3 — Use OpenStingerQwenAgent in Your Code

Replace your existing `qwen_agent.agents.Assistant` instantiation with `OpenStingerQwenAgent`. The wrapper is a thin subclass — all existing Qwen-Agent parameters pass through unchanged.

### Before (standard Qwen-Agent)

```python
from qwen_agent.agents import Assistant

agent = Assistant(
    llm={"model": "qwen-max", "api_key": "YOUR_KEY"},
    function_list=["web_search"],
    system_message="You are a helpful research assistant.",
)

messages = []
while True:
    user_input = input("User: ")
    messages.append({"role": "user", "content": user_input})
    for response in agent.run(messages):
        print(response)
    messages.extend(response)
```

### After (with OpenStinger)

```python
from openstinger.adapters.qwen_agent.agent import OpenStingerQwenAgent
import uuid

agent = OpenStingerQwenAgent(
    # OpenStinger parameters
    agent_id="my-research-agent",
    sessions_dir="/absolute/path/to/qwen_sessions",  # same as config.yaml sessions_dir
    session_id=f"session-{uuid.uuid4().hex[:8]}",    # unique per conversation
    inject_openstinger_skill=True,                   # adds tool guidance to system prompt

    # All existing Qwen-Agent parameters — unchanged
    llm={"model": "qwen-max", "api_key": "YOUR_KEY"},
    function_list=["web_search"],
    system_message="You are a helpful research assistant.",
)

messages = []
while True:
    user_input = input("User: ")
    messages.append({"role": "user", "content": user_input})
    for response in agent.run(messages):
        print(response)
    messages.extend(response)
```

**The only changes:**
- Import `OpenStingerQwenAgent` instead of `Assistant`
- Add `agent_id`, `sessions_dir`, `session_id`, `inject_openstinger_skill`
- Everything else — `llm`, `function_list`, `system_message`, `run()` — identical

### Connecting OpenStinger MCP Tools to Qwen-Agent

To give the agent access to OpenStinger's 30 tools (so it can call `memory_query`, `ops_status`, etc.), add the MCP server to Qwen-Agent's `function_list`:

```python
agent = OpenStingerQwenAgent(
    agent_id="my-research-agent",
    sessions_dir="/absolute/path/to/qwen_sessions",
    session_id=f"session-{uuid.uuid4().hex[:8]}",
    inject_openstinger_skill=True,

    llm={"model": "qwen-max", "api_key": "YOUR_KEY"},
    function_list=[
        "web_search",
        {
            "mcpServers": {
                "openstinger": {
                    "url": "http://localhost:8766/sse"
                }
            }
        }
    ],
    system_message="You are a helpful research assistant.",
)
```

> **⚠ IDE agent note:** Verify the exact format of the `mcpServers` block in `function_list` against the actual `qwen-agent>=0.0.10` API. The structure shown above is based on Qwen-Agent's documented MCP integration pattern — confirm against the library source or changelog before publishing.

---

## Step 4 — Session ID Strategy

Each `session_id` creates a separate JSONL file in `sessions_dir`. OpenStinger ingests all of them.

**Choose a session ID strategy that fits your use case:**

```python
import uuid
from datetime import date

# Option A: Unique per run (every script execution is a new session)
session_id = f"session-{uuid.uuid4().hex[:8]}"

# Option B: Date-based (all conversations on the same day share a file)
session_id = f"session-{date.today().isoformat()}"

# Option C: User-based (one file per user in multi-user deployments)
session_id = f"user-{user_id}"

# Option D: Fixed (one growing file — simplest for single-user personal agents)
session_id = "main"
```

For personal agents running on Mac Studio or a local machine, Option D (`session_id="main"`) is the simplest and mirrors OpenClaw's default session layout.

---

## Step 5 — Running with a Local Model (Mac Studio M4 Ultra)

OpenStinger works with any OpenAI-compatible endpoint. If you are running Qwen models locally via Ollama or vLLM on your Mac Studio M4 Ultra, configure both Qwen-Agent and OpenStinger to use the local endpoint.

### Qwen-Agent with local model

```python
agent = OpenStingerQwenAgent(
    agent_id="local-qwen-agent",
    sessions_dir="/absolute/path/to/qwen_sessions",
    session_id="main",
    inject_openstinger_skill=True,

    llm={
        "model": "qwen3.5-35b-a3b",          # or whatever tag you pulled
        "model_server": "http://localhost:11434/v1",  # Ollama endpoint
        "api_key": "ollama",                  # Ollama doesn't require a real key
    },
    function_list=[
        {
            "mcpServers": {
                "openstinger": {
                    "url": "http://localhost:8766/sse"
                }
            }
        }
    ],
    system_message="You are a helpful agent with persistent memory.",
)
```

### OpenStinger with local embeddings (optional)

```yaml
# config.yaml
llm:
  provider: openai                              # OpenAI-compatible
  model: deepseek/deepseek-v3.2               # any model via Novita or local
  llm_base_url: "https://api.novita.ai/v3/openai"
  embedding_model: qwen/qwen3-embedding-8b
  embedding_provider: openai
  embedding_base_url: "https://api.novita.ai/v3/openai"
```

This gives you a fully private stack: local Qwen inference on M4 Ultra + local OpenStinger memory on the same machine. Zero cloud dependency for the agent runtime. OpenStinger's LLM calls (entity extraction, dedup, classification) can point at the local endpoint too if you prefer.

---

## Verifying the Integration

### Check writer is creating session files

After a Qwen-Agent conversation:

```bash
ls /absolute/path/to/qwen_sessions/
# Expected: session-abc12345.jsonl (or whatever session_id you chose)

cat /absolute/path/to/qwen_sessions/session-abc12345.jsonl
# Expected: one JSON object per line, role: user or assistant
```

### Check ingestion is running

```bash
python -m openstinger.cli progress
```

After a few conversations:

```
Ingestion:  3 / 3 files processed   [████████████] 100%
Episodes:   42 in FalkorDB
Entities:   9 known
Vault:      identity: 0  domain: 2  (still building — needs ~1,000 episodes)
Gradient:   observe_only=true
```

### Check sessions appear in PostgreSQL

```bash
# In Adminer at http://localhost:8080
SELECT file_path, entries_read, last_read_at
FROM ingestion_cursor
WHERE agent_id = 'qwen-agent'
ORDER BY last_read_at DESC;
```

### Test semantic memory retrieval

In your next Qwen-Agent session, explicitly invoke memory:

```python
messages = [
    {"role": "user", "content": "Search your memory for anything we've discussed about research tasks."}
]
for response in agent.run(messages):
    print(response)
```

The agent should call `memory_query()` and return results from OpenStinger's temporal graph.

---

## What QwenSessionWriter Captures

The writer intercepts messages at the `run()` boundary:

| Message type | Captured? | Reason |
|---|---|---|
| `role: user` | ✅ Yes | Primary memory content |
| `role: assistant` (text) | ✅ Yes | Agent responses |
| `role: system` | ❌ No | Internal routing — not episodic content |
| `role: tool` (tool calls) | ❌ No | Operational noise — not useful for memory |
| `role: tool` (tool results) | ❌ No | Operational noise |
| Empty content | ❌ No | Nothing to store |
| Non-string content | ❌ No | Images, binary — not yet supported |

This mirrors the same filtering rules as OpenClaw's JSONL ingestion. The temporal graph receives clean episodic content only.

---

## Three Adoption Modes

### Mode 1 — Alongside (Start Here)

OpenStinger tools are available to the agent, writer captures sessions, ingestion runs. Qwen-Agent's own in-memory history (the `messages` list) still manages within-session context.

**When:** Day 1.  
**What to do:** Complete Steps 1–4. Run normally.  
**Benefit:** Zero behaviour change. OpenStinger builds memory in the background.

### Mode 2 — Primary

At the start of each new session, explicitly query OpenStinger for relevant context and inject it into the initial `messages` list before the first user turn.

```python
import httpx

# Before starting a session, retrieve relevant context from OpenStinger
def get_memory_context(query: str) -> str:
    # Call OpenStinger's memory_query via MCP HTTP or direct tool call
    # Return formatted context string
    ...

# Inject into session
context = get_memory_context("recent research tasks and preferences")
messages = [
    {
        "role": "system",
        "content": f"Relevant memory from previous sessions:\n{context}"
    }
]
```

**When:** After ~500–1,000 episodes are ingested.  
**Benefit:** Cross-session continuity — the agent remembers past work.

### Mode 3 — Exclusive

OpenStinger is the sole memory backend. The agent calls `memory_query()` at the start of every session and `memory_add()` for important facts during the session. The in-memory `messages` list manages only the current context window.

**When:** After Tier 2 vault has identity + domain notes (typically 2–4 weeks of daily use).  
**Benefit:** True persistent identity. The agent has a coherent, evolving self-model.

---

## Browser UIs

| URL | Tool | What it shows |
|---|---|---|
| `http://localhost:3000` | FalkorDB Browser | Visual graph: episodes, entities, relationships |
| `http://localhost:8080` | Adminer | PostgreSQL: all tables, ingestion jobs, alignment events |

**FalkorDB Browser login:** host `host.docker.internal` · port `6379` · leave password blank

**Adminer login:** System: PostgreSQL · Server: `host.docker.internal` · Username: `openstinger` · Password: `your_postgres_password` (value from your `.env`) · Database: `openstinger`

---

## Troubleshooting

### "No JSONL files created after running the agent"

- Confirm you are using `OpenStingerQwenAgent`, not the base `Assistant`
- Confirm `sessions_dir` in both the wrapper and `config.yaml` point to the same absolute path
- Confirm the directory exists and is writable: `mkdir -p /absolute/path/to/qwen_sessions`
- Check that the conversation included at least one `user` or `assistant` turn with non-empty string content

### "JSONL files exist but no episodes in FalkorDB"

- Confirm `session_format: qwen_agent` in OpenStinger `config.yaml`
- Confirm OpenStinger MCP server was restarted after changing `config.yaml`
- Check `.openstinger/openstinger.log` for ingestion errors
- Run `python -m openstinger.cli progress` — if files show 0 episodes, check the JSONL format with `cat`

### "Agent is not calling memory_query even though tools are connected"

- Confirm `inject_openstinger_skill=True` in `OpenStingerQwenAgent`
- The skill injection adds guidance to the system prompt — verify it appears with a debug print of `agent.system_message`
- Qwen-Agent only calls tools when it determines they are relevant — try explicitly asking "search your memory for..."

### "MCP connection error: Connection refused"

- Confirm OpenStinger MCP server is running on port 8766: `curl http://localhost:8766/sse`
- Confirm no firewall is blocking localhost connections
- If OpenStinger is on a different machine, update the URL in the `mcpServers` block

### "ImportError: cannot import name 'OpenStingerQwenAgent'"

- Confirm you installed with the `qwen-agent` extras: `pip install "openstinger[qwen-agent]"`
- Confirm `qwen-agent>=0.0.10` is installed: `pip show qwen-agent`
- Confirm you are in the correct virtual environment

### "Writer creates files but they are empty"

- The writer buffers up to 10 messages before flushing. Call `agent.writer.flush()` explicitly at the end of a session if you need immediate persistence
- Confirm the conversation had user/assistant turns — tool calls are not captured

---

## Comparison: Qwen-Agent + OpenStinger vs. Qwen-Agent Alone

| Capability | Qwen-Agent alone | With OpenStinger |
|---|---|---|
| Within-session memory | ✅ in-memory messages list | ✅ same |
| Cross-session memory | ❌ starts fresh every run | ✅ bi-temporal graph |
| Semantic retrieval | ❌ none | ✅ BM25 + vector hybrid |
| Entity coherence | ❌ no deduplication | ✅ 3-stage LSH + LLM |
| Conflict resolution | ❌ silent overwrite | ✅ temporal precedence |
| Structured self-knowledge | ❌ none | ✅ identity / domain / constraint vault |
| Alignment evaluation | ❌ none | ✅ 4-dimensional per-response scoring |
| Audit trail | ❌ none | ✅ 12-table PostgreSQL log |
| Memory portability | ❌ in-memory only | ✅ Docker volumes — any host |
| Local model support | ✅ Ollama / vLLM | ✅ same endpoints work for OpenStinger too |

---

## Reference — All 27 OpenStinger Tools

### Tier 1 — Memory (9 tools)

| Tool | When to call |
|---|---|
| `memory_add(content)` | Store an important fact explicitly |
| `memory_query(query, limit=5)` | Semantic recall — "what did we work on last sprint?" |
| `memory_search(query)` | Exact terms, IPs, numbers, wallet addresses |
| `memory_get_entity(entity_id)` | Fetch entity and its current relationships |
| `memory_get_episode(episode_id)` | Fetch a specific episode by ID |
| `memory_job_status()` | Check ingestion job queue |
| `memory_ingest_now()` | Trigger immediate ingestion cycle |
| `memory_namespace_status(agent_namespace)` | Episode/entity/edge counts |
| `memory_list_agents()` | List all registered namespaces |

### Tier 2 — StingerVault (11 additional tools, 20 total)

| Tool | When to call |
|---|---|
| `vault_status()` | Vault health and note counts |
| `vault_sync_now()` | Trigger immediate classification |
| `vault_stats()` | Note counts per category |
| `vault_promote_now(content, category)` | Promote a fact to structured knowledge |
| `vault_note_list(category)` | Browse: identity / domain / methodology / preference / constraint |
| `vault_note_get(note_id)` | Read a specific vault note |
| `vault_note_add(content, category)` | **v0.7** — manually seed a vault note (identity, constraint, etc.) |
| `knowledge_ingest(source, source_type)` | Ingest URL / PDF / YouTube / raw text |
| `namespace_create(agent_id)` | Create a new agent namespace |
| `namespace_list()` | List all namespaces |
| `namespace_archive(agent_id)` | Archive a namespace |

### Tier 3 — Gradient (8 additional tools, 28 total)

| Tool | When to call |
|---|---|
| `ops_status()` | **Start of every session** — single-call health dashboard |
| `gradient_status()` | Gradient health and observe_only flag |
| `gradient_alignment_score(content)` | Evaluate a response before sending |
| `gradient_drift_status()` | Rolling window alignment trend |
| `gradient_alignment_log(since_timestamp)` | Recent alignment evaluation events |
| `gradient_alert()` | Current drift alert status |
| `gradient_history(n=10)` | Last N verdicts with scores |
| `drift_status()` | Behavioral drift history from PostgreSQL |

---

*OpenStinger v0.7 · Qwen-Agent Integration Guide*  
*MIT License · https://github.com/srikanthbellary/openstinger*
