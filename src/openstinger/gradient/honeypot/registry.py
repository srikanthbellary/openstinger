"""
HoneypotRegistry — pattern registry for GradientHoneypot (v0.9).

Loads system-default patterns and operator-defined custom patterns from
GradientConfig.honeypot (populated from config.yaml gradient.honeypot block).

Pattern matching is **case-insensitive substring** — no regex, no embeddings.
This keeps the check sub-millisecond so it can fire on every retrieval call.

System defaults cover credential-adjacent terms, system-path probing, and
exfiltration-adjacent strings (sourced from the honeypot feature spec).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openstinger.config import HarnessConfig

logger = logging.getLogger(__name__)

# System-default patterns — sourced from spec §3
_DEFAULT_PATTERNS: list[str] = [
    # Credential-adjacent
    "api_key",
    "secret_key",
    "aws_credentials",
    "openai_key",
    "access_token",
    "auth_token",
    "password",
    "private_key",
    "ssh_key",
    # System path probing
    "system_auth",
    "etc_passwd",
    "shadow_file",
    "hosts_file",
    # Exfiltration-adjacent
    "env_file",
    "dotenv",
    "environment_variables",
    "config_secrets",
]


class HoneypotRegistry:
    """
    Holds the active set of honeypot patterns after merging defaults and custom config.

    Attributes:
        default_patterns: Active system-default patterns (after disabled_defaults removed).
        custom_patterns:  Operator-defined patterns from gradient.honeypot.custom_patterns.
    """

    def __init__(self, cfg: "HarnessConfig") -> None:
        honeypot_cfg = getattr(cfg.gradient, "honeypot", None)

        # System defaults minus any operator-disabled entries
        disabled: set[str] = set()
        custom: list[str] = []

        if honeypot_cfg is not None:
            disabled = {p.lower() for p in (honeypot_cfg.disabled_defaults or [])}
            custom   = list(honeypot_cfg.custom_patterns or [])

        self.default_patterns: list[str] = [
            p for p in _DEFAULT_PATTERNS if p.lower() not in disabled
        ]
        self.custom_patterns: list[str] = custom

        logger.info(
            "HoneypotRegistry loaded: %d default patterns, %d custom patterns",
            len(self.default_patterns),
            len(self.custom_patterns),
        )

    def all_patterns(self) -> list[tuple[str, str]]:
        """Return [(pattern, source), ...] for all active patterns."""
        result: list[tuple[str, str]] = [
            (p, "default") for p in self.default_patterns
        ] + [
            (p, "custom")  for p in self.custom_patterns
        ]
        return result

    @property
    def total_count(self) -> dict[str, int]:
        return {
            "default": len(self.default_patterns),
            "custom":  len(self.custom_patterns),
        }
