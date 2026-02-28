# OpenStinger + OpenClaw

OpenClaw is the origin of the \*Claw ecosystem — 207k+ GitHub stars, the largest community, and the most feature-complete agent runtime. OpenStinger was originally built alongside OpenClaw and has the deepest integration with it.

## What OpenStinger adds to OpenClaw

OpenClaw ships with QMD for search and Graphiti for temporal knowledge  both strong tools. OpenStinger adds what neither provides: structured operational observability, autonomous knowledge classification, entity governance, and behavioral alignment evaluation.

| Capability | OpenClaw (default) | OpenClaw + QMD | OpenStinger |
|---|---|---|---|
| Session search | SQLite FTS5 | BM25 + LLM reranking | Hybrid BM25 + vector + graph traversal |
| Entity extraction | None | None | 3-stage: string  MinHash LSH  LLM |
| Knowledge distillation | Manual (MEMORY.md) | Manual | Autonomous StingerVault classification |
| Operational audit trail | None | None | 12-table PostgreSQL schema |
| Alignment evaluation | Static SOUL.md | Static SOUL.md | Dynamic Gradient (per-output scoring) |
| Framework portability | OpenClaw-only | OpenClaw-only | Any MCP runtime |

## Setup

### 1. Start OpenStinger

```bash
docker compose up -d   # starts FalkorDB, PostgreSQL, Adminer
python -m openstinger.gradient.mcp.server
```

### 2. Configure mcporter in OpenClaw

In OpenClaw's `mcporter.json`:

```json
{
  "connections": {
    "openstinger": {
      "type": "sse",
      "url": "http://host.docker.internal:8766/sse"
    }
  }
}
```

> Note: Use `host.docker.internal` (not `localhost`) when connecting from inside the OpenClaw Docker container.

### 3. Configure session ingestion

OpenStinger reads OpenClaw's JSONL session files directly:

```yaml
# config.yaml
ingestion:
  sessions_dir: "/path/to/openclaw-data/config/agents/main/sessions"
  session_format: openclaw
```

### 4. Add OpenStinger to Claudia's SKILL.md

Add a `SKILL.md` to `openclaw-data/workspace/skills/openstinger/SKILL.md` so Claudia knows how to use all 27 tools. See [the OpenStinger repo](../README.md#mcp-tools) for the full tool reference.

## Operational Queries You Can Run Immediately

After setup, open Adminer (`http://localhost:8080`) and run:

```sql
-- How many episodes has Claudia processed this week?
SELECT DATE(TO_TIMESTAMP(created_at)) AS day, COUNT(*) AS episodes
FROM episode_log GROUP BY day ORDER BY day DESC LIMIT 7;

-- What entities does Claudia know the most about?
SELECT name, entity_type, episode_count
FROM entity_registry ORDER BY episode_count DESC LIMIT 20;

-- Vault health: active vs stale knowledge by category
SELECT category,
       COUNT(*) FILTER (WHERE stale = false) AS active,
       ROUND(AVG(confidence)::numeric, 2) AS avg_confidence
FROM vault_notes GROUP BY category;

-- Alignment verdict distribution this week
SELECT verdict, COUNT(*) AS count
FROM alignment_events
WHERE evaluated_at > EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days')
GROUP BY verdict;
```

---

*For the full OpenStinger setup guide, see the [main README](../README.md).*
