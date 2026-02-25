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

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to write config template")
    @click.option("--env", default=".env", help="Path to write .env template")
    def init(config: str, env: str) -> None:
        """Scaffold config templates and initialise FalkorDB schema."""
        config_path = Path(config)
        env_path = Path(env)

        if not config_path.exists():
            _write_config_template(config_path)
            click.echo(f"  config:  {config_path}")
        else:
            click.echo(f"  config:  {config_path} (already exists, skipped)")

        if not env_path.exists():
            _write_env_template(env_path)
            click.echo(f"  env:     {env_path}")
        else:
            click.echo(f"  env:     {env_path} (already exists, skipped)")

        click.echo("\nNext steps:")
        click.echo("  1. Fill in your API keys in .env")
        click.echo("  2. Start FalkorDB:  docker compose up -d")
        click.echo("  3. Run:             openstinger-cli health")
        click.echo("  4. Start server:    openstinger")

    # -----------------------------------------------------------------------
    # openstinger-cli health
    # -----------------------------------------------------------------------

    @cli.command()
    @click.option("--config", default="config.yaml", help="Path to config file")
    def health(config: str) -> None:
        """Check FalkorDB connectivity, operational DB, and configuration."""
        asyncio.run(_health_check(config))

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

    def _write_config_template(path: Path) -> None:
        path.write_text(
            "# OpenStinger configuration\n"
            "# Copy this file, fill in your values, and point OPENSTINGER_CONFIG to it.\n\n"
            "agent_name: my-agent\n"
            "agent_namespace: default\n\n"
            "falkordb:\n"
            "  host: localhost\n"
            "  port: 6379\n"
            "  password: ''\n\n"
            "operational_db:\n"
            "  provider: sqlite\n"
            "  sqlite_path: .openstinger/openstinger.db\n\n"
            "llm:\n"
            "  provider: anthropic\n"
            "  model: claude-sonnet-4-6\n"
            "  fast_model: claude-haiku-4-5-20251001\n"
            "  embedding_model: text-embedding-3-small\n"
            "  embedding_provider: openai\n\n"
            "mcp:\n"
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

    def _write_env_template(path: Path) -> None:
        path.write_text(
            "# OpenStinger environment variables\n"
            "ANTHROPIC_API_KEY=sk-ant-...\n"
            "OPENAI_API_KEY=sk-...    # for embeddings\n"
            "FALKORDB_HOST=localhost\n"
            "FALKORDB_PORT=6379\n"
            "FALKORDB_PASSWORD=\n"
            "OPENSTINGER_AGENT_NAME=default\n"
            "OPENSTINGER_LOG_LEVEL=INFO\n",
            encoding="utf-8",
        )
