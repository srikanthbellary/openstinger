"""Unit tests for token-budget packing (v0.10 wave 2)."""

import pytest

from openstinger.search.packer import pack_items, _count_tokens

pytestmark = pytest.mark.tier1


def test_pack_empty():
    packed, used, dropped = pack_items([], max_tokens=100)
    assert packed == []
    assert used == 0
    assert dropped == 0


def test_pack_none_budget_keeps_all():
    items = [
        {"uuid": "1", "content": "hello world"},
        {"uuid": "2", "content": "second"},
    ]
    packed, used, dropped = pack_items(items, max_tokens=None)
    assert len(packed) == 2
    assert dropped == 0
    assert used > 0


def test_pack_drops_tail_when_budget_tight():
    items = [
        {"uuid": "1", "content": "short"},
        {"uuid": "2", "content": "also short"},
        {"uuid": "3", "content": "third item here"},
    ]
    # Force tiny budget so only first fits after first item
    packed, used, dropped = pack_items(items, max_tokens=5)
    assert packed[0]["uuid"] == "1"
    assert len(packed) >= 1
    assert dropped == len(items) - len(packed)
    assert used <= 5 or packed[0].get("truncated")


def test_pack_always_includes_top_even_if_oversize():
    huge = "word " * 5000
    items = [{"uuid": "top", "content": huge}, {"uuid": "two", "content": "tiny"}]
    packed, used, dropped = pack_items(items, max_tokens=20)
    assert packed[0]["uuid"] == "top"
    assert packed[0].get("truncated") is True
    # Budget may still fit a tiny second item after truncation; top must remain first
    assert used > 0
    assert all(p["uuid"] != "missing" for p in packed)


def test_count_tokens_positive():
    assert _count_tokens("hello") >= 1
