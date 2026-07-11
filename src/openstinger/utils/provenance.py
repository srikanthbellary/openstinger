"""
Provenance utility — v0.9 hash-chained write provenance.

Provides two pure functions:
  compute_content_hash()  — SHA-256 of episode content string
  compute_receipt_hash()  — SHA-256 of a structured receipt (tamper-evident chain)

The chain works as follows:
  - Each write computes a receipt that includes the previous_hash (or "GENESIS" for first write).
  - The resulting provenance_hash is stored on the EpisodeLog row.
  - An auditor can replay writes in id-order and verify each hash links correctly.
"""

import hashlib
import json
import time


def compute_content_hash(content: str) -> str:
    """SHA-256 of the raw episode content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_receipt_hash(
    episode_uuid:    str,
    agent_namespace: str,
    content_hash:    str,
    valid_at:        float | int,
    previous_hash:   str | None,
) -> str:
    """
    SHA-256 of a structured receipt dictionary (sorted keys, deterministic encoding).

    The receipt includes:
      episode_uuid    — identifies the write event
      agent_namespace — namespace scope
      content_hash    — SHA-256 of the episode content (links content to chain)
      valid_at        — unix timestamp of the episode's temporal validity
      previous_hash   — hash of the preceding write, or "GENESIS" for the first write
      ts              — wall-clock time at receipt creation (monotonic tamper evidence)

    Note: ts is included to prevent pre-computation attacks.
    """
    receipt = {
        "episode_uuid":    episode_uuid,
        "agent_namespace": agent_namespace,
        "content_hash":    content_hash,
        "valid_at":        float(valid_at),
        "previous_hash":   previous_hash or "GENESIS",
        "ts":              time.time(),
    }
    return hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode("utf-8")
    ).hexdigest()
