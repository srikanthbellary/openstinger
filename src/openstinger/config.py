"""
OpenStinger configuration — HarnessConfig, resolve_path(), env loading.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Path resolution utility (Windows-safe, cross-platform)
# ---------------------------------------------------------------------------

def resolve_path(raw: str | Path | None, base: Path | None = None) -> Path | None:
    """
    Resolve a path string to an absolute Path, cross-platform safe.

    Steps:
      1. Expand environment variables  ($VAR or %VAR%)
      2. Expand ~ (user home)
      3. If relative and base provided, join with base
      4. Resolve to absolute path (no symlink resolution)

    Args:
        raw:  Raw path string or Path.  None → None.
        base: Optional base directory for relative paths.

    Returns:
        Absolute Path or None.
    """
    if raw is None:
        return None

    p = str(raw).strip()
    if not p:
        return None

    # Expand %WINVAR% style (Windows) and $VAR / ${VAR} style (Unix)
    p = os.path.expandvars(p)

    # Expand ~
    p = os.path.expanduser(p)

    path = Path(p)

    if not path.is_absolute():
        if base is not None:
            path = base / path
        else:
            path = Path.cwd() / path

    # Use resolve() only to normalise (collapse .., etc.) — don't follow symlinks strictly
    # strict=False means it won't raise if path doesn't exist yet
    return path.resolve()


def _expand_env_in_value(value: str) -> str:
    """Expand ${VAR:default} and ${VAR} patterns in YAML string values."""
    def replacer(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(2)  # may be None
        result = os.environ.get(var)
        if result is None:
            return default if default is not None else ""
        return result

    return re.sub(r"\$\{([^}:]+)(?::([^}]*))?\}", replacer, value)


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class FalkorDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    temporal_graph: str = "openstinger_temporal"
    knowledge_graph: str = "openstinger_knowledge"
    vector_dimensions: int = 1536

    @model_validator(mode="before")
    @classmethod
    def expand_env(cls, data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = _expand_env_in_value(v)
        return data


class OperationalDBConfig(BaseModel):
    provider: Literal["sqlite", "postgresql"] = "sqlite"
    sqlite_path: str = ".openstinger/openstinger.db"
    postgresql_url: Optional[str] = None

    @model_validator(mode="after")
    def validate_postgresql(self) -> "OperationalDBConfig":
        if self.provider == "postgresql" and not self.postgresql_url:
            raise ValueError("postgresql_url required when provider=postgresql")
        return self


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"
    llm_base_url: Optional[str] = None        # override for OpenAI-compatible providers
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "anthropic"] = "openai"
    embedding_base_url: Optional[str] = None  # override for non-OpenAI embedding providers


class IngestionConfig(BaseModel):
    sessions_dir: Optional[str] = None
    workspace_dir: Optional[str] = None
    poll_interval_seconds: int = 5
    chunk_size: int = 10
    session_format: str = "openclaw"  # "openclaw" | "simple"


class DeduplicationConfig(BaseModel):
    token_overlap_min: float = 0.4
    lsh_threshold: float = 0.5
    llm_confidence_min: float = 0.85
    identity_confidence_min: float = 0.92


class CommunityConfig(BaseModel):
    update_every_n: int = 0


class VaultConfig(BaseModel):
    vault_dir: str = ".openstinger/vault"
    classification_interval_seconds: int = 300
    sync_interval_seconds: int = 120
    decay_days: int = 90
    episodes_per_classification_batch: int = 20


class GradientConfig(BaseModel):
    enabled: bool = False
    observe_only: bool = True
    evaluation_timeout_ms: int = 2000
    drift_window_size: int = 20
    drift_alert_threshold: float = 0.65
    consecutive_flag_limit: int = 5
    min_outputs_before_active: int = 100


class MCPConfig(BaseModel):
    transport: Literal["stdio", "sse"] = "stdio"
    tcp_port: int = 8765


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Optional[str] = None


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class HarnessConfig(BaseModel):
    """Root configuration object for the OpenStinger harness."""

    agent_name: str = "default"
    agent_namespace: str = "default"

    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    operational_db: OperationalDBConfig = Field(default_factory=OperationalDBConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    gradient: GradientConfig = Field(default_factory=GradientConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime-resolved paths (set post-init, not from YAML)
    _root_dir: Optional[Path] = None

    @field_validator("agent_name", "agent_namespace")
    @classmethod
    def no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("agent_name and agent_namespace must not contain spaces")
        return v

    def resolved_sqlite_path(self, root: Path | None = None) -> Path:
        base = root or self._root_dir or Path.cwd()
        return resolve_path(self.operational_db.sqlite_path, base)  # type: ignore[return-value]

    def resolved_vault_dir(self, root: Path | None = None) -> Path:
        base = root or self._root_dir or Path.cwd()
        return resolve_path(self.vault.vault_dir, base)  # type: ignore[return-value]

    def resolved_sessions_dir(self, root: Path | None = None) -> Path | None:
        if not self.ingestion.sessions_dir:
            return None
        base = root or self._root_dir or Path.cwd()
        return resolve_path(self.ingestion.sessions_dir, base)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    env_file: str | Path | None = None,
    root_dir: Path | None = None,
) -> HarnessConfig:
    """
    Load HarnessConfig from YAML file + .env.

    Search order for config file:
      1. Explicit config_path argument
      2. OPENSTINGER_CONFIG env var
      3. ./config.yaml
      4. ~/.openstinger/config.yaml
      5. Defaults only

    Args:
        config_path: Explicit path to config.yaml
        env_file:    Explicit path to .env file (default: .env in cwd)
        root_dir:    Root directory for resolving relative paths in config
    """
    # Load .env first so env vars are available for config expansion
    env_path = resolve_path(env_file) if env_file else Path.cwd() / ".env"
    if env_path and env_path.exists():
        load_dotenv(env_path)

    # Find config file
    candidates: list[Path | None] = [
        resolve_path(config_path) if config_path else None,
        resolve_path(os.environ.get("OPENSTINGER_CONFIG", "")) or None,
        Path.cwd() / "config.yaml",
        Path.home() / ".openstinger" / "config.yaml",
    ]

    raw: dict = {}
    for candidate in candidates:
        if candidate and candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            break

    cfg = HarnessConfig.model_validate(raw)
    cfg._root_dir = root_dir or Path.cwd()
    return cfg
