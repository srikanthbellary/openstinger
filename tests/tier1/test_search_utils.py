"""Unit tests for BM25 / hybrid search helpers (v0.10)."""

from openstinger.temporal.search_utils import (
    apply_preference_boost,
    apply_recency_packaging,
    diversify_by_source,
    effective_search_limit,
    extract_search_terms,
    extract_subqueries,
    sanitize_bm25_query,
)


def test_sanitize_quotes_terms_and_drops_stopwords():
    q = sanitize_bm25_query("What was the last degree I graduated with?")
    assert '"degree"' in q
    assert '"graduated"' in q
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
    assert sanitize_bm25_query("a to of") == '""'


def test_diversify_by_source_prefers_new_sessions():
    rows = [
        {"uuid": "1", "source_description": "s1", "score": 1.0},
        {"uuid": "2", "source_description": "s1", "score": 0.9},
        {"uuid": "3", "source_description": "s2", "score": 0.5},
        {"uuid": "4", "source_description": "s3", "score": 0.4},
    ]
    out = diversify_by_source(rows, 3)
    sources = [r["source_description"] for r in out]
    assert sources[0] == "s1"
    assert "s2" in sources
    assert "s3" in sources


def test_preference_boost_on_recommend_query():
    rows = [
        {"uuid": "1", "content": "I prefer hotels with a rooftop pool.", "score": 0.5},
        {"uuid": "2", "content": "I went to the store yesterday.", "score": 0.8},
    ]
    out = apply_preference_boost(rows, "Can you suggest a hotel in Miami?")
    by_id = {r["uuid"]: r for r in out}
    assert by_id["1"]["score"] > 0.5
    assert by_id["1"].get("preference_boost") is True
    assert by_id["2"]["score"] == 0.8


def test_recency_labels_conflicts():
    rows = [
        {
            "uuid": "old",
            "content": "Rachel moved to Chicago last year.",
            "valid_at": 100,
            "valid_at_human": "Jan 2022",
            "score": 0.7,
        },
        {
            "uuid": "new",
            "content": "Rachel moved to the suburbs recently.",
            "valid_at": 200,
            "valid_at_human": "Jun 2023",
            "score": 0.7,
        },
    ]
    packed, conflicts = apply_recency_packaging(rows)
    by_id = {r["uuid"]: r for r in packed}
    assert by_id["old"].get("recency_label") == "older"
    assert by_id["new"].get("recency_label") == "newer"
    assert conflicts and conflicts[0]["entity_hint"] == "Rachel"


def test_extract_subqueries_includes_concrete_chunks():
    qs = extract_subqueries(
        "How many model kits have I worked on or bought?"
    )
    assert qs[0].startswith("How many model")
    blob = " ".join(qs).lower()
    assert "model" in blob and "kits" in blob


def test_extract_query_focus_terms():
    from openstinger.temporal.search_utils import extract_query_focus_terms
    terms = extract_query_focus_terms("Where did Rachel move after leaving Chicago?")
    assert "Rachel" in terms
    assert "Chicago" in terms
    miami = extract_query_focus_terms("Can you suggest a hotel for my trip to Miami?")
    assert "Miami" in miami


def test_episode_cues_bridge_store_and_coupon():
    from openstinger.temporal.search_utils import extract_episode_cues, package_episode_row
    text = (
        "I've been using the Cartwheel app from Target and it's helpful.\n"
        "user: I actually redeemed a $5 coupon on coffee creamer last Sunday."
    )
    cues = extract_episode_cues(text)
    assert any("Target" in s or "target" in s.lower() for s in cues["stores"])
    assert cues["summary"] and "link:" in cues["summary"]
    row = package_episode_row({"uuid": "1", "content": text, "score": 1.0}, query="Where did I redeem coupon?")
    assert "[Session cues]" in row["content"]
    assert "Target" in row["content"] or "target" in row["content"].lower()


def test_inventory_digest_for_count_queries():
    from openstinger.temporal.search_utils import build_inventory_digest
    hits = [
        {
            "source_description": "s1",
            "cues": {"actions": ["returned boots to Zara", "picked up exchanged boots"]},
            "content": "x",
        },
        {
            "source_description": "s2",
            "cues": {"actions": ["pick up navy blazer from dry cleaning"]},
            "content": "y",
        },
    ]
    dig = build_inventory_digest(hits, "How many clothing items do I need to pick up?")
    assert "RETURN" in dig and "PICKUP" in dig
    assert "Errand rows listed: 3" in dig
    assert "s1" in dig and "s2" in dig


def test_expertise_digest_scopes_recommend():
    from openstinger.temporal.search_utils import build_expertise_digest, apply_preference_boost
    deep = (
        "I've been working in the field of medical image analysis. "
        "MICCAI and BRATS radiology segmentation papers are central to my research."
    )
    shallow = (
        "In my statement of purpose I listed NeurIPS ICML CVPR and environmental "
        "sustainability for a graduate program application."
    )
    hits = [
        {"uuid": "deep", "content": deep, "score": 0.4},
        {"uuid": "shallow", "content": shallow, "score": 0.9},
    ]
    boosted = apply_preference_boost(hits, "Can you recommend publications I might find interesting?")
    by_id = {r["uuid"]: r for r in boosted}
    assert by_id["deep"]["score"] > by_id["shallow"]["score"]
    dig = build_expertise_digest(boosted, "recommend publications interesting")
    assert "specialty" in dig.lower()
    assert "medical" in dig.lower() or "imaging" in dig.lower()
    assert "environmental" not in dig.lower()

def test_location_update_prefers_moved_language():
    from openstinger.temporal.search_utils import prioritize_query_entity_recency
    rows = [
        {"uuid": "noise", "content": "Rachel likes pizza.", "valid_at": 300, "score": 0.9},
        {"uuid": "old", "content": "Rachel moved to Chicago.", "valid_at": 100, "score": 0.7},
        {"uuid": "new", "content": "Rachel moved to the suburbs recently.", "valid_at": 200, "score": 0.7},
    ]
    out = prioritize_query_entity_recency(rows, "Where did Rachel move to after her relocation?")
    assert out[0]["uuid"] == "new"
    assert out[0].get("entity_latest") is True


def test_smart_excerpt_keeps_query_terms():
    from openstinger.temporal.search_utils import smart_episode_excerpt
    body = ("pad " * 3000) + "I redeemed a $5 coupon on coffee creamer at Target." + (" tail " * 3000)
    out = smart_episode_excerpt(body, "Where did I redeem a $5 coupon on coffee creamer?", max_chars=2000)
    assert "Target" in out
    assert "coupon" in out.lower() or "creamer" in out.lower()
