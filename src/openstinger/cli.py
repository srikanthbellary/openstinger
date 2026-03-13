"""
OpenStinger CLI — developer and ops tooling.

Subcommands:
    openstinger-cli init      — scaffold config templates + FalkorDB schema
    openstinger-cli health    — check FalkorDB connectivity + DB status
    openstinger-cli migrate   — apply DB schema migrations
    openstinger-cli db schema — print current FalkorDB graph schemas
    openstinger-cli vault     — vault inspection commands
    openstinger-cli cache     — embedding cache management

Install:
    pip install -e ".[dev]"

Usage:
    openstinger-cli health
    openstinger-cli init --config ./config.yaml
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import click
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False


def _require_click() -> None:
    if not _HAS_CLICK:
        print("ERROR: click is required for the CLI: pip install click", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point guard
# ---------------------------------------------------------------------------

def main() -> None:
    _require_click()
    cli()  # type: ignore[call-arg]


if _HAS_CLICK:
    import click

    @click.group()
    @click.version_option()
    def cli() -> None:
        """OpenStinger CLI — developer and operations tooling."""

    # -----------------------------------------------------------------------
    # openstinger-cli init
    # -----------------------------------------------------------------------

    # Known Ollama embedding models with their native output dimensions
    _OLLAMA_MODEL_DIMS: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to write config template")
    @click.option("--env", default=".env", help="Path to write .env")
    @click.option("--provider", type=click.Choice(["anthropic", "openai", "novita", "deepseek"]),
                  default=None, help="LLM provider")
    @click.option("--api-key", default=None, help="API key (prompted if omitted)")
    @click.option("--agent-name", default=None, help="Agent name (e.g. claudia)")
    @click.option("--namespace", default=None, help="Agent namespace (e.g. claudia)")
    @click.option("--profile-dir", default=None, multiple=True,
                  help="Path(s) to agent workspace dirs for identity ingestion (v0.7)")
    def init(
        config: str,
        env: str,
        provider: str | None,
        api_key: str | None,
        agent_name: str | None,
        namespace: str | None,
        profile_dir: tuple[str, ...],
    ) -> None:
        """Set up OpenStinger: generate .env with secure passwords, scaffold config."""
        import secrets

        click.echo("\nOpenStinger Setup")
        click.echo("=" * 40)

        # --- Agent identity ---
        if agent_name is None:
            agent_name = click.prompt("Agent name", default="my-agent")
        if namespace is None:
            namespace = click.prompt("Agent namespace", default=agent_name)

        # --- LLM Provider ---
        if provider is None:
            provider = click.prompt(
                "LLM provider",
                type=click.Choice(["anthropic", "openai", "novita", "deepseek"]),
                default="anthropic",
            )

        # --- API key ---
        if api_key is None:
            key_label = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            api_key = click.prompt(f"{key_label}", hide_input=True, default="", show_default=False)

        # --- Embedding provider (v0.8: Ollama option) ---
        embed_provider = click.prompt(
            "\nEmbedding provider",
            type=click.Choice(["cloud", "ollama"]),
            default="cloud",
        )
        ollama_host = "http://localhost:11434"
        embedding_model = "text-embedding-3-small"
        vector_dimensions = 1536

        if embed_provider == "ollama":
            ollama_host = click.prompt("Ollama host", default="http://localhost:11434")
            preset_choice = click.prompt(
                "Embedding model",
                type=click.Choice(["nomic-embed-text", "mxbai-embed-large", "all-minilm", "custom"]),
                default="nomic-embed-text",
            )
            if preset_choice == "custom":
                embedding_model = click.prompt("Model name")
                vector_dimensions = click.prompt("Output dimensions", type=int)
            else:
                embedding_model = preset_choice
                vector_dimensions = _OLLAMA_MODEL_DIMS[preset_choice]

            click.echo(f"\n  Ollama embeddings: {embedding_model}  ({vector_dimensions}d)")
            click.echo("  WARNING: If you change embedding models later, you must wipe and")
            click.echo("  re-create the FalkorDB indices (vector dimensions are not compatible")
            click.echo("  across models). Run: docker compose down -v && docker compose up -d")
        else:
            embedding_model = "text-embedding-3-small"
            vector_dimensions = 1536

        # --- Profile dirs (v0.7 AgentProfileIngester) ---
        profile_dirs: list[str] = list(profile_dir)
        if not profile_dirs:
            click.echo("\nProfile directories (v0.7): OpenStinger will scan these paths for")
            click.echo("identity files (SKILL.md, SOUL.md, AGENTS.md, etc.) and auto-seed")
            click.echo("the vault with your agent's identity on boot.")
            raw = click.prompt(
                "Profile dir(s) [space-separated, or Enter to skip]",
                default="",
                show_default=False,
            )
            if raw.strip():
                profile_dirs = raw.strip().split()

        # --- Generate passwords ---
        falkordb_pass = secrets.token_urlsafe(24)
        postgres_pass = secrets.token_urlsafe(24)

        # --- Write .env ---
        env_path = Path(env)
        if not env_path.exists():
            _write_env_file(env_path, provider, api_key, falkordb_pass, postgres_pass, embed_provider)
            click.echo(f"\n  .env     {env_path}  ✓  (passwords auto-generated)")
        else:
            click.echo(f"\n  .env     {env_path}  (already exists — skipped)")

        # --- Write config ---
        config_path = Path(config)
        if not config_path.exists():
            _write_config_template(
                config_path, agent_name, namespace, profile_dirs,
                embed_provider=embed_provider,
                embedding_model=embedding_model,
                vector_dimensions=vector_dimensions,
                ollama_host=ollama_host,
            )
            click.echo(f"  config   {config_path}  ✓")
        else:
            click.echo(f"  config   {config_path}  (already exists — skipped)")

        click.echo("\n✅  Ready.")
        click.echo("   Next: docker compose up -d")
        click.echo("         openstinger-cli health")
        click.echo("         openstinger")
        click.echo("")

    # -----------------------------------------------------------------------
    # openstinger-cli health
    # -----------------------------------------------------------------------

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to config file")
    def health(config: str) -> None:
        """Check FalkorDB connectivity, operational DB, and configuration."""
        asyncio.run(_health_check(config))

    # -----------------------------------------------------------------------
    # openstinger-cli progress
    # -----------------------------------------------------------------------

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to config file")
    def progress(config: str) -> None:
        """Show ingestion progress, vault notes, and gradient readiness."""
        asyncio.run(_progress(config))

    async def _progress(config_path: str) -> None:
        import os, sqlite3, json
        from openstinger.config import load_config
        from openstinger.temporal.falkordb_driver import wait_for_falkordb

        cfg = load_config(config_path if Path(config_path).exists() else None)
        db_path = cfg.resolved_sqlite_path()

        click.echo("=" * 52)
        click.echo(" OpenStinger v0.5 — Live Progress")
        click.echo("=" * 52)

        # ── Ingestion cursor progress ──────────────────────────────────────
        sessions_dir = cfg.resolved_sessions_dir()
        total_bytes = 0
        consumed_bytes = 0
        files_started = 0
        files_done = 0

        if sessions_dir and Path(str(sessions_dir)).exists():
            all_files = list(Path(str(sessions_dir)).glob("*.jsonl"))
            total_bytes = sum(f.stat().st_size for f in all_files)

            try:
                conn = sqlite3.connect(str(db_path))
                row = conn.execute(
                    "SELECT session_file_cursor_json FROM session_state WHERE agent_namespace = ?",
                    (cfg.agent_namespace,),
                ).fetchone()
                conn.close()
                cursors: dict = json.loads(row[0] or "{}") if row else {}
                consumed_bytes = sum(v for v in cursors.values() if isinstance(v, int) and v > 0)
                files_started = sum(1 for v in cursors.values() if isinstance(v, int) and v > 0)
                files_done = sum(
                    1 for fname, pos in cursors.items()
                    if isinstance(pos, int) and pos > 0
                    and (sessions_dir / Path(fname).name).exists()
                    and pos >= (sessions_dir / Path(fname).name).stat().st_size
                )
            except Exception as exc:
                click.echo(f"  (cursor read error: {exc})")

        pct = (consumed_bytes / total_bytes * 100) if total_bytes else 0
        click.echo(f"\nIngestion (namespace={cfg.agent_namespace}):")
        click.echo(f"  Files:    {files_done} done / {files_started} started / {len(all_files) if sessions_dir else '?'} total")
        click.echo(f"  Bytes:    {consumed_bytes/1024/1024:.1f} MB / {total_bytes/1024/1024:.1f} MB  ({pct:.1f}%)")

        # ── FalkorDB live counts ───────────────────────────────────────────
        try:
            driver = await wait_for_falkordb(
                cfg.falkordb.host, cfg.falkordb.port, cfg.falkordb.password,
                timeout_seconds=5.0,
            )
            await driver.init_schema()
            rows = await driver.query_temporal("MATCH (ep:Episode) RETURN count(ep) AS n")
            episodes = rows[0].get("n", 0) if rows else 0
            rows = await driver.query_temporal("MATCH (e:Entity) RETURN count(e) AS n")
            entities = rows[0].get("n", 0) if rows else 0
            rows = await driver.query_temporal("MATCH ()-[r:RELATES_TO]-() RETURN count(r) AS n")
            facts = rows[0].get("n", 0) if rows else 0

            # Vault notes
            cats = ["identity", "domain", "methodology", "preference", "constraint"]
            note_counts: dict[str, int] = {}
            for cat in cats:
                cat_rows = await driver.query_knowledge(
                    "MATCH (n:Note {agent_namespace: $ns, category: $cat}) "
                    "WHERE n.stale = 0 RETURN count(n) AS n",
                    {"ns": cfg.agent_namespace, "cat": cat},
                )
                note_counts[cat] = cat_rows[0].get("n", 0) if cat_rows else 0
            total_notes = sum(note_counts.values())

            await driver.close()

            click.echo(f"\nFalkorDB (namespace={cfg.agent_namespace}):")
            click.echo(f"  Episodes: {episodes:,}")
            click.echo(f"  Entities: {entities:,}")
            click.echo(f"  Facts:    {facts:,}")

            click.echo(f"\nVault notes (total active={total_notes}):")
            for cat, n in note_counts.items():
                bar = "█" * min(n, 20)
                click.echo(f"  {cat:15s} {n:3d}  {bar}")

            # ── Gradient readiness ─────────────────────────────────────────
            identity_ok = note_counts.get("identity", 0) >= 15
            constraint_ok = note_counts.get("constraint", 0) >= 10
            click.echo(f"\nGradient activation readiness:")
            click.echo(f"  identity notes ≥ 15:    {'✅' if identity_ok else '❌'}  (have {note_counts.get('identity',0)})")
            click.echo(f"  constraint notes ≥ 10:  {'✅' if constraint_ok else '❌'}  (have {note_counts.get('constraint',0)})")
            if identity_ok and constraint_ok:
                click.echo("  → READY: set gradient.observe_only: false in config.yaml, then restart.")
            else:
                need_id = max(0, 15 - note_counts.get("identity", 0))
                need_co = max(0, 10 - note_counts.get("constraint", 0))
                click.echo(f"  → Not ready yet. Need {need_id} more identity, {need_co} more constraint notes.")
                click.echo("     Vault builds notes automatically as more episodes are ingested.")
        except Exception as exc:
            click.echo(f"  FalkorDB unavailable: {exc}")

        # ── Embedding cache ────────────────────────────────────────────────
        try:
            from openstinger.storage.embedding_cache import EmbeddingCache
            cache_path = db_path.parent / "embed_cache.db"
            if cache_path.exists():
                ec = EmbeddingCache(cache_path, cfg.llm.embedding_model)
                stats = await ec.stats()
                click.echo(f"\nEmbedding cache: {stats.get('total_entries',0)} entries, {stats.get('total_hits',0)} hits")
        except Exception:
            pass

        click.echo("=" * 52)


    async def _health_check(config_path: str) -> None:
        from openstinger.config import load_config

        click.echo("OpenStinger Health Check")
        click.echo("=" * 40)

        # Config
        try:
            cfg = load_config(config_path if Path(config_path).exists() else None)
            click.echo(f"  config:        OK  (namespace={cfg.agent_namespace})")
        except Exception as exc:
            click.echo(f"  config:        FAIL  ({exc})")
            return

        # FalkorDB
        try:
            from openstinger.temporal.falkordb_driver import wait_for_falkordb
            driver = await wait_for_falkordb(
                cfg.falkordb.host, cfg.falkordb.port,
                cfg.falkordb.password, timeout_seconds=5.0,
            )
            ok = await driver.ping()
            await driver.close()
            click.echo(f"  falkordb:      {'OK' if ok else 'FAIL'}  ({cfg.falkordb.host}:{cfg.falkordb.port})")
        except Exception as exc:
            click.echo(f"  falkordb:      FAIL  ({exc})")

        # Operational DB
        try:
            from openstinger.operational.adapter import create_adapter
            db_path = cfg.resolved_db_path() if hasattr(cfg, "resolved_db_path") else None
            db = create_adapter(
                provider=cfg.operational_db.provider,
                sqlite_path=db_path,
                postgresql_url=getattr(cfg.operational_db, "postgresql_url", None),
            )
            await db.init()
            await db.close()
            click.echo(f"  operational_db: OK  ({cfg.operational_db.provider})")
        except Exception as exc:
            click.echo(f"  operational_db: FAIL  ({exc})")

        # LLM API key
        import os
        key_name = "ANTHROPIC_API_KEY" if cfg.llm.provider == "anthropic" else "OPENAI_API_KEY"
        has_key = bool(os.environ.get(key_name))
        click.echo(f"  llm_api_key:   {'OK' if has_key else 'MISSING'}  ({key_name})")

        embed_key = bool(os.environ.get("OPENAI_API_KEY"))
        click.echo(f"  embed_api_key: {'OK' if embed_key else 'MISSING'}  (OPENAI_API_KEY)")

    # -----------------------------------------------------------------------
    # openstinger-cli migrate
    # -----------------------------------------------------------------------

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to config file")
    def migrate(config: str) -> None:
        """Apply DB schema migrations (create_all — idempotent)."""
        asyncio.run(_migrate(config))

    async def _migrate(config_path: str) -> None:
        from openstinger.config import load_config
        from openstinger.operational.adapter import create_adapter

        try:
            cfg = load_config(config_path if Path(config_path).exists() else None)
            db_path = cfg.resolved_db_path() if hasattr(cfg, "resolved_db_path") else None
            db = create_adapter(
                provider=cfg.operational_db.provider,
                sqlite_path=db_path,
                postgresql_url=getattr(cfg.operational_db, "postgresql_url", None),
            )
            await db.init()
            await db.close()
            click.echo(f"Migration complete ({cfg.operational_db.provider})")
        except Exception as exc:
            click.echo(f"Migration failed: {exc}", err=True)
            sys.exit(1)

    # -----------------------------------------------------------------------
    # openstinger-cli db
    # -----------------------------------------------------------------------

    @cli.group()
    def db() -> None:
        """Database inspection commands."""

    @db.command("schema")
    @click.option("--config", default="config.yaml", help="Path to config file")
    def db_schema(config: str) -> None:
        """Print current FalkorDB graph schemas."""
        asyncio.run(_db_schema(config))

    async def _db_schema(config_path: str) -> None:
        from openstinger.config import load_config
        from openstinger.temporal.falkordb_driver import wait_for_falkordb

        cfg = load_config(config_path if Path(config_path).exists() else None)
        driver = await wait_for_falkordb(
            cfg.falkordb.host, cfg.falkordb.port, cfg.falkordb.password,
        )
        await driver.connect()
        await driver.init_schema()

        for graph_name, query_fn in [
            ("temporal", driver.query_temporal),
            ("knowledge", driver.query_knowledge),
        ]:
            click.echo(f"\n{graph_name.upper()} GRAPH ({driver.temporal_graph_name if graph_name == 'temporal' else driver.knowledge_graph_name}):")
            try:
                rows = await query_fn("CALL db.indexes()")
                for row in rows:
                    click.echo(f"  {json.dumps(row, default=str)}")
            except Exception as exc:
                click.echo(f"  (error: {exc})")

        await driver.close()

    # -----------------------------------------------------------------------
    # openstinger-cli vault
    # -----------------------------------------------------------------------

    @cli.group()
    def vault() -> None:
        """Vault inspection commands."""

    @vault.command("status")
    @click.option("--config", default="config.yaml", help="Path to config file")
    def vault_status(config: str) -> None:
        """Show vault stats (note counts by category)."""
        asyncio.run(_vault_status(config))

    @vault.command("import")
    @click.option("--config", default="config.yaml", help="Path to config file")
    @click.option("--dir", "source_dir", required=True, help="Directory containing .md files to import")
    @click.option("--recursive", is_flag=True, default=False, help="Recurse into subdirectories")
    @click.option("--category", default=None, help="Override category for all imported notes (default: infer from subdir)")
    def vault_import(config: str, source_dir: str, recursive: bool, category: Optional[str]) -> None:
        """Bulk import .md files into the vault (idempotent — skips unchanged files)."""
        asyncio.run(_vault_import(config, source_dir, recursive, category))

    async def _vault_import(config_path: str, source_dir: str, recursive: bool, category_override: Optional[str]) -> None:
        from openstinger.config import load_config
        from openstinger.temporal.falkordb_driver import wait_for_falkordb
        from openstinger.temporal.openai_embedder import OpenAIEmbedder
        from openstinger.storage.embedding_cache import CachedEmbedder, EmbeddingCache
        from openstinger.operational.adapter import create_adapter
        from openstinger.scaffold.vault_engine import VaultEngine

        cfg = load_config(config_path if Path(config_path).exists() else None)
        source = Path(source_dir).expanduser().resolve()

        if not source.exists():
            click.echo(f"ERROR: directory not found: {source}", err=True)
            return

        pattern = "**/*.md" if recursive else "*.md"
        md_files = sorted(source.glob(pattern))

        if not md_files:
            click.echo(f"No .md files found in {source}")
            return

        click.echo(f"Found {len(md_files)} .md file(s) in {source}")

        driver = await wait_for_falkordb(
            cfg.falkordb.host, cfg.falkordb.port, cfg.falkordb.password,
            vector_dimensions=cfg.falkordb.vector_dimensions,
        )
        await driver.init_schema()

        db = create_adapter(
            provider=cfg.operational_db.provider,
            sqlite_path=cfg.resolved_sqlite_path(),
            postgresql_url=cfg.operational_db.postgresql_url,
        )
        await db.init()

        if cfg.llm.embedding_provider == "ollama":
            raw_embedder = OpenAIEmbedder(
                api_key="ollama",
                model=cfg.llm.embedding_model,
                dimensions=cfg.falkordb.vector_dimensions,
                base_url=f"{cfg.llm.ollama_host}/v1",
                skip_dimensions=True,
            )
        else:
            raw_embedder = OpenAIEmbedder(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=cfg.llm.embedding_model,
                dimensions=cfg.falkordb.vector_dimensions,
                base_url=cfg.llm.embedding_base_url or None,
            )

        _cache_db = cfg.resolved_sqlite_path().parent / "embed_cache.db"
        _embed_cache = EmbeddingCache(db_path=_cache_db, model_name=cfg.llm.embedding_model)
        await _embed_cache.init()
        embedder = CachedEmbedder(embedder=raw_embedder, cache=_embed_cache)

        vault_dir = cfg.resolved_vault_dir()
        vault_engine = VaultEngine(
            driver=driver,
            db=db,
            embedder=embedder,
            agent_namespace=cfg.agent_namespace,
            vault_dir=vault_dir,
        )

        imported = 0
        skipped = 0
        errors = 0

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if not content:
                    skipped += 1
                    continue

                if category_override:
                    cat = category_override
                else:
                    rel = md_file.relative_to(source)
                    cat = rel.parts[0] if len(rel.parts) > 1 else "notes"
                    known = {"identity", "domain", "methodology", "preference", "constraint", "notes"}
                    if cat not in known:
                        cat = "notes"

                existing_hash = await db.get_vault_checksum(str(md_file))
                import hashlib
                current_hash = hashlib.sha256(content.encode()).hexdigest()

                if existing_hash and existing_hash == current_hash:
                    skipped += 1
                    continue

                await vault_engine._create_note({
                    "category": cat,
                    "content": content,
                    "confidence": 0.85,
                })
                await db.upsert_vault_checksum(str(md_file), current_hash)
                imported += 1
                click.echo(f"  [{cat}] {md_file.name}")
            except Exception as exc:
                errors += 1
                click.echo(f"  ERROR: {md_file.name} — {exc}", err=True)

        await driver.close()
        await db.close()

        click.echo(f"\nDone: {imported} imported, {skipped} skipped, {errors} errors")

    async def _vault_status(config_path: str) -> None:
        from openstinger.config import load_config
        from openstinger.temporal.falkordb_driver import wait_for_falkordb

        cfg = load_config(config_path if Path(config_path).exists() else None)
        driver = await wait_for_falkordb(
            cfg.falkordb.host, cfg.falkordb.port, cfg.falkordb.password,
        )
        await driver.init_schema()

        categories = ["identity", "domain", "methodology", "preference", "constraint"]
        click.echo(f"Vault stats (namespace={cfg.agent_namespace}):")
        for cat in categories:
            rows = await driver.query_knowledge(
                """
                MATCH (n:Note {agent_namespace: $ns, category: $cat})
                RETURN sum(CASE WHEN n.stale = 0 THEN 1 ELSE 0 END) AS active,
                       sum(CASE WHEN n.stale = 1 THEN 1 ELSE 0 END) AS stale
                """,
                {"ns": cfg.agent_namespace, "cat": cat},
            )
            stats = rows[0] if rows else {"active": 0, "stale": 0}
            click.echo(f"  {cat:15s} active={stats.get('active', 0)}  stale={stats.get('stale', 0)}")

        await driver.close()

    # -----------------------------------------------------------------------
    # openstinger-cli cache
    # -----------------------------------------------------------------------

    @cli.group()
    def cache() -> None:
        """Embedding cache management."""

    @cache.command("stats")
    @click.option("--config", default="config.yaml", help="Path to config file")
    def cache_stats(config: str) -> None:
        """Show embedding cache statistics."""
        asyncio.run(_cache_stats(config))

    async def _cache_stats(config_path: str) -> None:
        from openstinger.config import load_config
        from openstinger.storage.embedding_cache import EmbeddingCache

        cfg = load_config(config_path if Path(config_path).exists() else None)
        db_path = cfg.resolved_db_path() if hasattr(cfg, "resolved_db_path") else Path(".openstinger/openstinger.db")
        cache_path = Path(str(db_path)).parent / "embed_cache.db"
        cache = EmbeddingCache(cache_path, cfg.llm.embedding_model)
        stats = await cache.stats()
        click.echo(f"Embedding cache:")
        for k, v in stats.items():
            click.echo(f"  {k}: {v}")

    @cache.command("clear")
    @click.option("--config", default="config.yaml", help="Path to config file")
    @click.confirmation_option(prompt="Clear all cached embeddings?")
    def cache_clear(config: str) -> None:
        """Clear the embedding cache for the configured model."""
        asyncio.run(_cache_clear(config))

    async def _cache_clear(config_path: str) -> None:
        from openstinger.config import load_config
        from openstinger.storage.embedding_cache import EmbeddingCache

        cfg = load_config(config_path if Path(config_path).exists() else None)
        db_path = cfg.resolved_db_path() if hasattr(cfg, "resolved_db_path") else Path(".openstinger/openstinger.db")
        cache_path = Path(str(db_path)).parent / "embed_cache.db"
        cache = EmbeddingCache(cache_path, cfg.llm.embedding_model)
        deleted = await cache.clear()
        click.echo(f"Cleared {deleted} cached embeddings")


    # -----------------------------------------------------------------------
    # Templates
    # -----------------------------------------------------------------------

    def _write_config_template(
        path: Path,
        agent_name: str = "my-agent",
        agent_namespace: str = "default",
        profile_dirs: list[str] | None = None,
        embed_provider: str = "cloud",
        embedding_model: str = "text-embedding-3-small",
        vector_dimensions: int = 1536,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        profile_dirs = profile_dirs or []
        if profile_dirs:
            profile_dirs_block = "ingestion:\n  profile_dirs:\n" + "".join(
                f"    - {d}\n" for d in profile_dirs
            )
        else:
            profile_dirs_block = (
                "# ingestion:\n"
                "#   profile_dirs:\n"
                "#     - /path/to/your/agent/workspace   "
                "# v0.7: auto-seeds vault from SKILL.md, SOUL.md, AGENTS.md etc.\n"
            )

        if embed_provider == "ollama":
            llm_embed_block = (
                f"  embedding_model: {embedding_model}\n"
                "  embedding_provider: ollama\n"
                f"  ollama_host: {ollama_host}\n"
            )
            falkordb_dims_line = f"  vector_dimensions: {vector_dimensions}\n"
        else:
            llm_embed_block = (
                f"  embedding_model: {embedding_model}\n"
                "  embedding_provider: openai\n"
            )
            falkordb_dims_line = f"  vector_dimensions: {vector_dimensions}\n"

        path.write_text(
            "# OpenStinger configuration\n"
            "# Generated by: openstinger-cli init\n\n"
            f"agent_name: {agent_name}\n"
            f"agent_namespace: {agent_namespace}\n\n"
            + profile_dirs_block
            + "\nfalkordb:\n"
            "  host: localhost\n"
            "  port: 6379\n"
            "  password: ''\n"
            + falkordb_dims_line
            + "\noperational_db:\n"
            "  provider: sqlite\n"
            "  sqlite_path: .openstinger/openstinger.db\n\n"
            "llm:\n"
            "  provider: anthropic\n"
            "  model: claude-sonnet-4-6\n"
            "  fast_model: claude-haiku-4-5-20251001\n"
            + llm_embed_block
            + "\nmcp:\n"
            "  transport: stdio\n"
            "  tcp_port: 8765\n\n"
            "vault:\n"
            "  vault_dir: .openstinger/vault\n"
            "  classification_interval_seconds: 300\n"
            "  sync_interval_seconds: 120\n\n"
            "gradient:\n"
            "  enabled: false\n"
            "  observe_only: true\n",
            encoding="utf-8",
        )

    def _write_env_file(
        path: Path,
        provider: str,
        api_key: str,
        falkordb_pass: str,
        postgres_pass: str,
        embed_provider: str = "cloud",
    ) -> None:
        """Write a fully populated .env with real generated passwords."""
        anthropic_key = api_key if provider == "anthropic" else ""
        openai_key = api_key if provider in ("openai", "novita", "deepseek") else ""
        if embed_provider == "ollama":
            embed_note = "# Embeddings: Ollama (local) — no API key needed for embeddings\n"
            openai_embed_line = "OPENAI_API_KEY=ollama    # placeholder required by client; Ollama ignores it\n"
        else:
            embed_note = ""
            openai_embed_line = f"OPENAI_API_KEY={openai_key}    # required for cloud embedding model\n"
        path.write_text(
            "# OpenStinger environment — generated by: openstinger-cli init\n"
            "# Do not commit this file.\n\n"
            "# --- LLM ---\n"
            f"ANTHROPIC_API_KEY={anthropic_key}\n"
            + embed_note
            + openai_embed_line
            + "\n# --- FalkorDB (graph + vector store) ---\n"
            "FALKORDB_HOST=localhost\n"
            "FALKORDB_PORT=6379\n"
            f"FALKORDB_PASSWORD={falkordb_pass}\n\n"
            "# --- PostgreSQL (operational audit DB) ---\n"
            f"POSTGRES_PASSWORD={postgres_pass}\n\n"
            "# --- Agent ---\n"
            "OPENSTINGER_AGENT_NAME=default\n"
            "OPENSTINGER_LOG_LEVEL=INFO\n"
            "OPENSTINGER_MCP_TRANSPORT=stdio\n",
            encoding="utf-8",
        )

    def _write_env_template(path: Path) -> None:
        """Legacy: write a blank template (use _write_env_file for interactive init)."""
        path.write_text(
            "# OpenStinger environment variables\n"
            "ANTHROPIC_API_KEY=\n"
            "OPENAI_API_KEY=    # for embeddings\n"
            "FALKORDB_HOST=localhost\n"
            "FALKORDB_PORT=6379\n"
            "FALKORDB_PASSWORD=\n"
            "POSTGRES_PASSWORD=\n"
            "OPENSTINGER_AGENT_NAME=default\n"
            "OPENSTINGER_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
