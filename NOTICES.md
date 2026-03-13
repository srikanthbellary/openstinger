# Notices & Acknowledgements

OpenStinger is an original implementation by Srikanth Bellary, licensed under MIT.  
No source code from any external project has been copied into this repository.  
The following projects influenced OpenStinger's architectural thinking.

---

## Architectural Inspiration

### Graphiti
**Repository:** https://github.com/getzep/graphiti  
**License:** Apache 2.0  
**Influence:** Bi-temporal knowledge graph concepts for agent memory — the idea of separating
*when events happened* (`valid_at`) from *when the agent learned them* (`recorded_at`).
OpenStinger implements this pattern independently on FalkorDB rather than Neo4j.

---

### Cognee
**Repository:** https://github.com/topoteretes/cognee  
**License:** Apache 2.0  
**Influence:** Semantic knowledge graph construction and the concept of distilling episodic
memory into structured, categorized self-knowledge. OpenStinger's StingerVault classification
pipeline is an independent implementation of this general approach.

---

### Zep
**Repository:** https://github.com/getzep/zep  
**License:** Apache 2.0  
**Influence:** Long-term memory architecture patterns for LLM applications, and the operational
model of running memory infrastructure as a service alongside agents rather than embedded
within them.

---

### Mem0
**Repository:** https://github.com/mem0ai/mem0  
**License:** Apache 2.0  
**Influence:** Hybrid vector + structured memory approaches, and the user/agent memory
separation model that informed OpenStinger's multi-agent namespace design.

---

### OpenClaw (OpenClaw)
**Repository:** https://github.com/openclaw/openclaw  
**License:** MIT  
**Influence:** Session JSONL format and the agent gateway architecture that OpenStinger
was originally built to extend. OpenStinger's byte-offset cursor ingestion pipeline
is designed around OpenClaw's session file structure.

---

## Runtime Dependencies

OpenStinger's runtime dependencies are listed in `pyproject.toml` and `requirements.txt`.
Each dependency retains its own license. Key infrastructure dependencies:

| Dependency | License | Purpose |
|---|---|---|
| FalkorDB | Server Side Public License (SSPL) | Temporal graph + vector store |
| PostgreSQL | PostgreSQL License (MIT-like) | Operational audit database |
| SQLAlchemy | MIT | ORM for PostgreSQL access |
| APScheduler | MIT | Vault classification scheduler |
| anthropic | MIT | LLM API client (entity extraction, dedup) |
| openai | MIT | Embeddings API client |
| mcp | MIT | Model Context Protocol server |
| datasketch | MIT | MinHash LSH for entity deduplication |
| asyncpg | Apache 2.0 | Async PostgreSQL driver |

Full dependency licenses can be inspected by running:
```bash
pip-licenses --with-urls --format=markdown
```

---

## A Note on Ideas

The agent memory space is a fast-moving field built on shared ideas. Temporal graphs,
entity extraction, confidence scoring, and episodic-to-semantic promotion are patterns
that multiple teams arrived at independently and in parallel. OpenStinger builds on
this collective thinking while contributing its own approaches: the 3-stage entity
deduplication pipeline, the StingerVault 5-category classification system, the
PostgreSQL operational audit schema, and the Gradient inference-time alignment harness.

We stand on the shoulders of the teams who figured out that agents need memory in the
first place.

---

*OpenStinger — MIT License — Copyright (c) 2026 Srikanth Bellary*
