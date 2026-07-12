"""Unit tests for BM25 / hybrid search helpers (v0.10)."""

from openstinger.temporal.search_utils import extract_search_terms, sanitize_bm25_query


def test_sanitize_quotes_terms_and_drops_stopwords():
    q = sanitize_bm25_query("What was the last degree I graduated with?")
    assert '"degree"' in q
    assert '"graduated"' in q
    assert "last" not in q.replace('"last"', "")  # stopword dropped
    assert ":" not in q


def test_sanitize_strips_redi_search_ops():
    q = sanitize_bm25_query('find "quoted" (ops) and foo:bar')
    assert "(" not in q
    assert ")" not in q
    assert ":" not in q
    assert '"foo"' in q or '"bar"' in q


def test_extract_search_terms_filters_short_and_stop():
    terms = extract_search_terms("How did I do on the exam in Miami?")
    assert "miami" in terms
    assert "exam" in terms
    assert "how" not in terms
    assert "the" not in terms


def test_sanitize_empty_fallback():
    assert sanitize_bm25_query("") == '""'
    # Only stopwords → quoted-empty fallback
    assert sanitize_bm25_query("a to of") == '""'
