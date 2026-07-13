"""Lexicon quarantine and experimental expand tests."""

from pathlib import Path

import pytest

from openstinger.search.experimental_lexicons import (
    KIT_EXPAND,
    PUB_EXPAND,
    ERRAND_EXPAND,
    DOMAIN_LEXICONS,
    append_experimental_subqueries,
)
from openstinger.temporal.search_utils import extract_subqueries
from openstinger.config import RetrievalConfig

pytestmark = pytest.mark.tier1

_SRC = Path(__file__).resolve().parents[2] / "src" / "openstinger"


def test_retrieval_config_lexicons_default_off():
    assert RetrievalConfig().experimental_lexicons is False
    assert RetrievalConfig().reranker == "none"


def test_experimental_appends_kit_and_pub_phrases():
    out: list[str] = []
    append_experimental_subqueries("recommend publications that are interesting", out.append)
    blob = " ".join(out).lower()
    assert "conference" in blob or "journal" in blob

    out2: list[str] = []
    append_experimental_subqueries("how many model kits did I buy", out2.append)
    blob2 = " ".join(out2).lower()
    assert any(k in blob2 for k in ("revell", "tamiya", "model kit", "scale model"))


def test_default_subqueries_exclude_quarantined_lists():
    qs = extract_subqueries("Can you recommend publications I might find interesting?")
    blob = " ".join(qs).lower()
    for banned in ("miccai", "brats", "neurips", "proceedings"):
        assert banned not in blob


def test_pipeline_module_has_no_hardcoded_rooftop_zara_in_source():
    """Default-path modules must not embed quarantine strings (benchmark integrity)."""
    pipeline = (_SRC / "search" / "pipeline.py").read_text(encoding="utf-8")
    # Allowed only via experimental import path, not literals
    assert "rooftop" not in pipeline.lower()
    assert "zara" not in pipeline.lower()
    assert "skip the basics" not in pipeline.lower()


def test_experimental_module_holds_quarantined_vocab():
    assert "revell" in " ".join(KIT_EXPAND).lower()
    assert "publication" in " ".join(PUB_EXPAND).lower()
    assert "dry cleaning" in " ".join(ERRAND_EXPAND).lower()
    assert "healthcare_imaging" in DOMAIN_LEXICONS
