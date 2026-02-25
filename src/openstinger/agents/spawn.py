"""
SpawnContracts — validation of agent configuration at startup.

A SpawnContract defines the minimum requirements an agent must satisfy
before it is allowed to start. Validates HarnessConfig against contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpawnContract:
    """
    Defines requirements for an agent to start successfully.

    Fields:
        requires_falkordb:   FalkorDB must be reachable.
        requires_llm_key:    LLM API key must be set.
        requires_embed_key:  Embedding API key must be set.
        min_tier:            Minimum tier to validate (1, 2, or 3).
        allowed_namespaces:  If non-empty, namespace must be in this list.
    """
    requires_falkordb: bool = True
    requires_llm_key: bool = True
    requires_embed_key: bool = True
    min_tier: int = 1
    allowed_namespaces: list[str] = field(default_factory=list)


@dataclass
class SpawnValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


async def validate_config(
    cfg: Any,
    contract: SpawnContract | None = None,
) -> SpawnValidationResult:
    """
    Validate a HarnessConfig against a SpawnContract.

    Args:
        cfg:      HarnessConfig instance.
        contract: SpawnContract to validate against (uses sensible defaults if None).

    Returns:
        SpawnValidationResult with valid=True if all checks pass.
    """
    if contract is None:
        contract = SpawnContract()

    errors: list[str] = []
    warnings: list[str] = []

    # Namespace validation
    if contract.allowed_namespaces and cfg.agent_namespace not in contract.allowed_namespaces:
        errors.append(
            f"Namespace '{cfg.agent_namespace}' not in allowed list: {contract.allowed_namespaces}"
        )

    # LLM key validation
    if contract.requires_llm_key:
        llm = cfg.llm
        if llm.provider == "anthropic" and not getattr(llm, "api_key", None):
            import os
            if not os.environ.get("ANTHROPIC_API_KEY"):
                errors.append("ANTHROPIC_API_KEY is not set and llm.api_key is empty")
        elif llm.provider == "openai" and not getattr(llm, "api_key", None):
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                errors.append("OPENAI_API_KEY is not set and llm.api_key is empty")

    # Embedding key validation
    if contract.requires_embed_key:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            warnings.append(
                "OPENAI_API_KEY not set — embedding may fail unless llm.llm_base_url is configured"
            )

    # Tier 2 validations
    if contract.min_tier >= 2:
        vault_dir = getattr(cfg, "vault", None)
        if vault_dir is None:
            errors.append("Tier 2 requires vault config (missing vault section in config)")

    # Tier 3 validations
    if contract.min_tier >= 3:
        gradient = getattr(cfg, "gradient", None)
        if gradient is None:
            errors.append("Tier 3 requires gradient config (missing gradient section in config)")

    valid = len(errors) == 0
    if not valid:
        for err in errors:
            logger.error("SpawnContract violation: %s", err)
    for warn in warnings:
        logger.warning("SpawnContract warning: %s", warn)

    return SpawnValidationResult(valid=valid, errors=errors, warnings=warnings)
