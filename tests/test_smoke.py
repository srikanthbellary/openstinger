"""
Smoke tests — basic import and instantiation checks.
No FalkorDB required. Validates the project wires together.
"""

from __future__ import annotations

import pytest


def test_import_config():
    from openstinger.config import HarnessConfig, load_config, resolve_path
    cfg = HarnessConfig()
    assert cfg.agent_name == "default"
    assert cfg.agent_namespace == "default"


def test_resolve_path_absolute():
    from pathlib import Path
    from openstinger.config import resolve_path
    p = resolve_path("/tmp/test")
    assert p is not None
    assert p.is_absolute()


def test_resolve_path_none():
    from openstinger.config import resolve_path
    assert resolve_path(None) is None


def test_import_temporal_models():
    from openstinger.temporal.nodes import EntityNode, EpisodeNode
    from openstinger.temporal.edges import EntityEdge, EpisodicEdge

    ep = EpisodeNode(content="test episode")
    assert ep.uuid is not None
    assert ep.content == "test episode"

    entity = EntityNode(name="Alice Smith", entity_type="PERSON")
    assert entity.name == "Alice Smith"


def test_entity_edge_expiry():
    from openstinger.temporal.edges import EntityEdge
    import uuid
    edge = EntityEdge(
        source_node_uuid=str(uuid.uuid4()),
        target_node_uuid=str(uuid.uuid4()),
        relation_type="WORKS_AT",
        fact="Alice works at Acme",
    )
    assert edge.is_current is True
    edge.expire()
    assert edge.is_current is False
    assert edge.expired_at is not None


def test_import_deduplicator():
    from openstinger.temporal.deduplicator import DeduplicationEngine, normalize_name
    assert normalize_name("Alice Smith") == "alice smith"
    assert normalize_name("Dr. Bob Jones") == "bob jones"
    assert normalize_name("Acme Corp") in ("acme", "acme corp")  # suffix stripped


def test_import_operational_models():
    from openstinger.operational.models import (
        IngestionJob, EpisodeLog, EntityRegistryRow, SessionState
    )
    job = IngestionJob(agent_namespace="test", source_file=None)
    assert job.status == "pending"
    job.mark_running()
    assert job.status == "running"


def test_import_gradient_profile():
    from openstinger.gradient.alignment_profile import AlignmentProfile
    profile = AlignmentProfile(state="empty")
    assert not profile.is_usable
    assert "no identity profile" in profile.identity_context()


def test_harness_config_validation():
    from openstinger.config import HarnessConfig
    import pytest
    with pytest.raises(Exception):
        HarnessConfig(agent_name="has spaces")


def test_config_sqlite_path_resolution():
    from openstinger.config import HarnessConfig
    from pathlib import Path
    cfg = HarnessConfig()
    path = cfg.resolved_sqlite_path(root=Path("/tmp"))
    assert path is not None
    assert str(path).endswith("openstinger.db")


def test_normalizer_unicode():
    from openstinger.temporal.deduplicator import normalize_name
    assert normalize_name("Müller") == normalize_name("Muller")


def test_episode_node_cypher_props():
    from openstinger.temporal.nodes import EpisodeNode
    ep = EpisodeNode(content="test", source="conversation")
    props = ep.to_cypher_props()
    assert "uuid" in props
    assert props["content"] == "test"
    assert isinstance(props["created_at"], int)
