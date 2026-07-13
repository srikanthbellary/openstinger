"""
v0.10 regression smoke against live FalkorDB / production graphs.

Non-destructive:
  - Uses isolated agent_namespace v010_regress_*
  - Deletes only that namespace at end
  - Counts claudia (or configured) namespace before/after; must be unchanged

Does not call LME. Does not touch Postgres schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from openstinger.config import load_config
from openstinger.mcp.tools.memory_tools import memory_query, memory_search
from openstinger.operational.adapter import SQLiteAdapter
from openstinger.temporal.engine import TemporalEngine, _MONTH_NAMES
from openstinger.temporal.entity_registry import EntityRegistry
from openstinger.temporal.falkordb_driver import FalkorDBDriver
from openstinger.temporal.nodes import EpisodeNode
from openstinger.temporal.openai_compatible_client import OpenAICompatibleClient
from openstinger.temporal.openai_embedder import OpenAIEmbedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("v010_regress")


async def count_episodes(driver: FalkorDBDriver, namespace: str) -> int:
    rows = await driver.query_temporal(
        "MATCH (ep:Episode {agent_namespace: $ns}) RETURN count(ep) AS c",
        {"ns": namespace},
    )
    return int(rows[0]["c"]) if rows else 0


async def main() -> int:
    cfg = load_config(ROOT / "config.yaml", env_file=ROOT / ".env", root_dir=ROOT)
    live_ns = cfg.agent_namespace or "claudia"
    test_ns = f"v010_regress_{uuid.uuid4().hex[:8]}"
    marker = f"v010-regress-marker-{uuid.uuid4().hex[:12]}"

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NOVITA_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY / NOVITA_API_KEY missing")
        return 2

    dims = cfg.falkordb.vector_dimensions
    driver = FalkorDBDriver(
        host=cfg.falkordb.host,
        port=cfg.falkordb.port,
        password=cfg.falkordb.password or "",
        temporal_graph_name=cfg.falkordb.temporal_graph,
        knowledge_graph_name=cfg.falkordb.knowledge_graph,
        vector_dimensions=dims,
    )
    await driver.connect()

    print("=== A: schema init + dim probe ===")
    await driver.init_schema()
    print(f"OK schema init (vector_dimensions={dims})")

    before = await count_episodes(driver, live_ns)
    print(f"=== B: live namespace '{live_ns}' episode count before = {before} ===")

    db_path = ROOT / ".openstinger" / f"{test_ns}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SQLiteAdapter(db_path)
    await db.init()

    llm = OpenAICompatibleClient(
        api_key=api_key,
        model=cfg.llm.model,
        fast_model=cfg.llm.fast_model,
        base_url=cfg.llm.llm_base_url,
    )
    # Match live config dims; skip_dimensions if native model size differs
    embedder = OpenAIEmbedder(
        api_key=api_key,
        model=cfg.llm.embedding_model,
        dimensions=dims,
        base_url=cfg.llm.embedding_base_url or cfg.llm.llm_base_url,
        skip_dimensions=(dims >= 4096),
    )
    registry = EntityRegistry(db)
    await registry.warmup()
    engine = TemporalEngine(
        driver=driver,
        llm=llm,
        embedder=embedder,
        entity_registry=registry,
        agent_namespace=test_ns,
        retrieval_config=cfg.retrieval,
        extract_statements=False,  # smoke write is raw persist; skip extra LLM
    )

    print(f"=== C: write isolated episode in {test_ns} ===")
    now = int(time.time())
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    month = _MONTH_NAMES[dt.month]
    episode = EpisodeNode(
        content=(
            f"Regression smoke note. Unique token {marker}. "
            f"User prefers concise answers about OpenStinger search packaging."
        ),
        source="message",
        source_description="v010_regression_smoke",
        agent_namespace=test_ns,
        valid_at=now,
    )
    episode.valid_at_human = f"{month} {dt.day} {dt.year} {month} {dt.year} {dt.year}"
    emb = await embedder.embed(episode.content)
    episode.content_embedding = emb
    await engine._persist_episode(episode)
    print(f"OK persisted uuid={episode.uuid} emb_dims={len(emb)}")

    print("=== D: query_memory fair fields (S1-S3) ===")
    t0 = time.time()
    qm = await engine.query_memory(query=f"What about {marker} OpenStinger?", limit=5)
    elapsed_qm = round(time.time() - t0, 3)
    eps = qm.get("episodes") or []
    ranked = qm.get("ranked") or []
    hit = next((e for e in eps if marker in (e.get("content") or "")), None)
    if hit is None:
        hit = next((e for e in ranked if marker in (e.get("content") or "")), None)
    assert hit is not None, f"marker not found in query_memory; episodes={len(eps)} ranked={len(ranked)}"
    assert hit.get("source_description") == "v010_regression_smoke", hit
    assert hit.get("valid_at_human"), f"missing valid_at_human: {hit}"
    assert qm.get("bm25_query"), "missing bm25_query echo"
    assert "retrieval_confidence" in qm, "missing retrieval_confidence"
    assert "pipeline" in qm and "channels_run" in (qm.get("pipeline") or {}), qm.get("pipeline")
    assert "abstain_suggested" in qm
    print(
        f"OK query_memory hit score={hit.get('score')} "
        f"source={hit.get('source_description')!r} "
        f"when={hit.get('valid_at_human')!r} "
        f"bm25={qm.get('bm25_query')!r} "
        f"conf={qm.get('retrieval_confidence')} "
        f"channels={qm.get('pipeline', {}).get('channels_run')} "
        f"elapsed={elapsed_qm}s"
    )

    print("=== E: memory_query / memory_search tool wrappers ===")
    mq = await memory_query(engine, query=marker, limit=5)
    assert any(marker in (e.get("content") or "") for e in mq.get("episodes") or []), mq
    assert "bm25_query" in mq
    ms = await memory_search(engine, query=marker, search_type="episodes", limit=5)
    assert any(marker in (e.get("content") or "") for e in ms.get("episodes") or []), ms
    # S1 fields on search rows when present
    ms_hit = next(e for e in ms["episodes"] if marker in (e.get("content") or ""))
    assert ms_hit.get("source_description") == "v010_regression_smoke"
    print("OK memory_query + memory_search")

    print("=== F: live namespace unchanged ===")
    after = await count_episodes(driver, live_ns)
    assert after == before, f"live ns count changed {before} -> {after}"
    print(f"OK '{live_ns}' still {after}")

    print("=== G: cleanup test namespace ===")
    await driver.query_temporal(
        "MATCH (n {agent_namespace: $ns}) DETACH DELETE n",
        {"ns": test_ns},
    )
    left = await count_episodes(driver, test_ns)
    assert left == 0, left
    await db.close()
    await driver.close()
    try:
        db_path.unlink(missing_ok=True)
    except OSError:
        pass
    print("OK cleaned up")
    print("RESULT: PASS v0.10 live regression smoke")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        logging.exception("RESULT: FAIL %s", exc)
        raise SystemExit(1)
