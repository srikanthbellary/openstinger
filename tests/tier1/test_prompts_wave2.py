"""Prompt builder tests for reflect and statement extraction."""

import pytest

from openstinger.temporal.prompts.reflect import REFLECT_SYSTEM, build_reflect_user
from openstinger.temporal.prompts.extraction import (
    build_extract_edges_user,
    build_extract_statements_user,
    EXTRACT_STATEMENTS_SYSTEM,
)

pytestmark = pytest.mark.tier1


def test_reflect_system_requires_insufficient_sentinel():
    assert "INSUFFICIENT_MEMORY" in REFLECT_SYSTEM


def test_build_reflect_user_includes_conflicts_and_blocks():
    user = build_reflect_user(
        "Where does Rachel live?",
        ["[1] sess\nRachel moved to the suburbs."],
        conflicts=[
            {
                "entity_hint": "Rachel",
                "older_valid_at_human": "2022",
                "newer_valid_at_human": "2023",
            }
        ],
    )
    assert "Where does Rachel live?" in user
    assert "Rachel moved" in user
    assert "Conflict hints" in user
    assert "Rachel" in user
    assert "Answer:" in user


def test_build_reflect_user_empty_memories():
    user = build_reflect_user("q", [])
    assert "(no memories)" in user


def test_statements_user_includes_reference_time():
    user = build_extract_statements_user("I got promoted last Friday", reference_time="2026-03-10")
    assert "2026-03-10" in user
    assert "promoted" in user
    assert "atomic" in EXTRACT_STATEMENTS_SYSTEM.lower() or "statement" in EXTRACT_STATEMENTS_SYSTEM.lower()


def test_edges_user_includes_reference_time():
    user = build_extract_edges_user(
        "Alice works at Acme",
        ["Alice", "Acme"],
        reference_time="2026-03-10",
    )
    assert "2026-03-10" in user
    assert "Alice" in user
