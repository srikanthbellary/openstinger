# Agent Skills Template for OpenStinger

A runtime-agnostic guide for teaching any MCP-compatible agent how to use OpenStinger's 28 tools effectively. Copy and adapt this into your agent's system prompt, skills file, or tool-use instructions.

---

## Overview

OpenStinger exposes 28 tools across three tiers. Each tier is additive — Tier 1 works standalone, Tier 2 builds on Tier 1, Tier 3 requires Tier 2 data to be useful.

| Tier | Name | Tools | When it becomes useful |
|---|---|---|---|
| Tier 1 | Memory Harness | 9 | Immediately on first session |
| Tier 2 | StingerVault | 11 | After ~500 episodes ingested |
| Tier 3 | Gradient | 8 | After ~50+ vault notes exist |

---

## Tier 1 — Memory Harness (9 tools)

### `memory_query` ⭐ (use this most)
**When:** At the start of any new session, or when the user references a past event, person, decision, or context you don't have in your current window.
```
memory_query(query="<topic or question>")
memory_query(query="trading bot crash", after_date="2026-01", before_date="2026-03")
```
Returns: episodes (raw events), entities (people/orgs/concepts), facts (relationships), ranked (combined).

### `memory_search`
**When:** `memory_query` returns nothing, or the query involves: IP addresses, prices, wallet addresses, partial names (typos), specific months/years.
```
memory_search(query="203.0.113.42")           # IP → CONTAINS fallback
memory_search(query="Qinn", search_type="entities")  # typo → vector fallback
memory_search(query="February 2026", search_type="episodes")
```

### `memory_add`
**When:** User shares something important mid-session that isn't in a session file yet — a decision, a key fact, a preference stated explicitly.
```
memory_add(content="User decided to sunset the trading bot on 2026-03-15", source="manual")
```

### `memory_ingest_now`
**When:** User asks you to "remember this session" or you want to trigger ingestion of recent session files immediately (normally runs on a poll interval).

### `memory_namespace_status`
**When:** Checking how much memory has been built up for this agent (episode count, entity count, session count).

### `memory_job_status` / `memory_get_entity` / `memory_get_episode` / `memory_list_agents`
**When:** Debugging — checking ingestion job status, fetching specific entities or episodes by UUID, listing all registered agent namespaces.

---

## Tier 2 — StingerVault (+11 tools, 20 total)

Vault is structured self-knowledge distilled from episodic memory. Five categories: **IDENTITY**, **DOMAIN**, **METHODOLOGY**, **PREFERENCE**, **CONSTRAINT**.

### `vault_note_get` / `vault_notes_list`
**When:** You want to know what the agent knows about itself — its identity, values, expertise areas, constraints, or preferences.
```
vault_notes_list(category="IDENTITY")     # who this agent is
vault_notes_list(category="CONSTRAINT")   # what it won't do
vault_notes_list(category="METHODOLOGY")  # how it approaches problems
```

### `vault_promote_now`
**When:** User asks you to summarize or consolidate recent sessions into structured notes. Triggers immediate vault classification (normally runs on schedule).

### `vault_note_create` / `vault_note_update` / `vault_note_expire`
**When:** Manually managing vault notes — creating explicit beliefs, updating stale ones, or retracting outdated notes.

### External document ingestion
```
ingest_url(url="https://...")         # webpage
ingest_pdf(path="/path/to/doc.pdf")   # PDF
ingest_youtube(url="https://youtu.be/...") # YouTube transcript
```
**When:** User wants you to read and remember an external resource permanently.

### `ops_status` ⭐
**When:** User asks "how is my agent doing?" or you need a single-call health dashboard.
Returns: vault note counts by category, gradient pass rate (7d), drift alert, memory totals.

---

## Tier 3 — Gradient (8 tools · 28 total across all tiers)

Gradient evaluates alignment. It starts in **`observe_only` mode** — it measures without intervening. Check readiness with `openstinger-cli progress` before activating.

### `gradient_alignment_score`
**When:** You want to evaluate a specific response text against the agent's alignment profile.
```
gradient_alignment_score(text="<response to evaluate>")
# Returns: score (0–1), verdict (pass/soft_flag/hard_block), dimensions
```

### `gradient_status`
**When:** Checking if Gradient is active, what the profile state is, whether observe_only is on.

### `gradient_drift_status` / `gradient_alert`
**When:** Checking for behavioral drift — is the agent's alignment score trending down over time?

### `gradient_history` ⭐
**When:** Reviewing the last N alignment evaluations — scores, verdicts, latency — from PostgreSQL.
```
gradient_history(limit=20)   # returns structured rows from alignment_events table
```

### `drift_status` ⭐
**When:** Checking the rolling window of behavioral drift metrics.

### `gradient_alignment_log`
**When:** Debugging specific evaluation events — what triggered a soft_flag or hard_block.

---

## Tool Chaining Patterns

### Pattern 1: Session start (recommended)
```
1. ops_status()                                → health check
2. vault_notes_list(category="IDENTITY")       → who am I in this context
3. memory_query(query="recent work context",
               after_date="<14 days ago>")     → what was I last doing
```
> **Why `after_date`?** Without it, semantic search may surface older sessions
> with higher word overlap over your actual recent work. Use `after_date` with a
> date 7–14 days ago to anchor the query to recent history.

### Pattern 2: User references something from the past
```
1. memory_query(query="<what they mentioned>")
   → if no results:
2. memory_search(query="<same query>")        → fuzzy + fallback
```

### Pattern 3: After a long session
```
1. memory_ingest_now()                        → ingest current session
2. vault_promote_now()                        → distill into vault notes
3. ops_status()                               → confirm
```

### Pattern 4: Enterprise audit query (PostgreSQL)
```
-- Outside the agent, connect BI tool to PostgreSQL:
SELECT verdict, score, latency_ms, evaluated_at
FROM alignment_events
WHERE agent_namespace = 'default'
ORDER BY evaluated_at DESC LIMIT 100;
```

---

## Empty Vault Handling

If `vault_notes_list` returns `count: 0` with a `hint` field, the vault has not been seeded yet. This is normal on first run.

**Do NOT retry the same tool in a loop.** Instead:
```
# The vault is empty — fall back to memory_query for context
memory_query(query="agent identity", limit=3)

# Or seed it manually with what you know:
vault_note_add(category="identity", content="<agent identity statement>")
```

---

## Docker Networking Note

OpenStinger MCP runs on the **host machine** at port 8766 (not in Docker). If your agent framework runs in Docker:

```yaml
# docker-compose.yml — ALL service containers that need MCP access must have:
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Then use MCP URL: `http://host.docker.internal:8766/sse`

Without `extra_hosts`, containers cannot reach the host MCP server.

---

## Quick Reference

| Goal | Tool |
|---|---|
| Find past events | `memory_query` |
| Find an entity by partial/fuzzy name | `memory_search(search_type="entities")` |
| Check who this agent is | `vault_notes_list(category="IDENTITY")` |
| Full health dashboard | `ops_status` |
| See recent alignment scores | `gradient_history` |
| Trigger ingestion now | `memory_ingest_now` |
| Read an external resource | `ingest_url` / `ingest_pdf` / `ingest_youtube` |

---

*For framework-specific wiring (OpenClaw, DeerFlow, Qwen-Agent, LangGraph), see the per-framework guides in this directory.*
