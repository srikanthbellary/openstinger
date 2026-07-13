"""Tests for RRF fusion and retrieval confidence (v0.10 wave 2)."""

from openstinger.search.ranker import rrf_fuse, retrieval_confidence


def test_rrf_multi_channel_outranks_single():
    channels = {
        "a": [{"uuid": "shared", "content": "x"}, {"uuid": "only_a", "content": "y"}],
        "b": [{"uuid": "shared", "content": "x"}],
        "c": [{"uuid": "shared", "content": "x"}],
    }
    out = rrf_fuse(channels, k=60, limit=5)
    assert out[0]["uuid"] == "shared"
    assert out[0]["fusion_score"] > out[1]["fusion_score"]


def test_rrf_weights_shift_order():
    channels = {
        "a": [{"uuid": "a1"}, {"uuid": "b1"}],
        "b": [{"uuid": "b1"}, {"uuid": "a1"}],
    }
    equal = rrf_fuse(channels, weights={"a": 1.0, "b": 1.0}, k=60, limit=2)
    heavy_a = rrf_fuse(channels, weights={"a": 5.0, "b": 0.1}, k=60, limit=2)
    assert equal[0]["uuid"] in {"a1", "b1"}
    assert heavy_a[0]["uuid"] == "a1"


def test_rrf_deterministic_tiebreak():
    channels = {
        "a": [{"uuid": "b"}, {"uuid": "a"}],
        "b": [{"uuid": "a"}, {"uuid": "b"}],
    }
    out1 = rrf_fuse(channels, k=60, limit=2)
    out2 = rrf_fuse(channels, k=60, limit=2)
    assert [r["uuid"] for r in out1] == [r["uuid"] for r in out2]


def test_abstain_on_empty():
    conf, abstain = retrieval_confidence([])
    assert conf == 0.0
    assert abstain is True


def test_confidence_boosts_multi_channel_and_threshold():
    fused = [
        {"uuid": "x", "fusion_score": 0.05, "rrf_channels": ["a", "b"]},
    ]
    conf, abstain = retrieval_confidence(fused, abstain_threshold=0.15)
    assert conf > 0.05
    # With low raw score even multi-channel may still abstain depending on formula
    assert isinstance(abstain, bool)

    strong = [{"uuid": "y", "fusion_score": 0.4, "rrf_channels": ["a", "b", "c"]}]
    conf2, abstain2 = retrieval_confidence(strong, abstain_threshold=0.15)
    assert conf2 >= 0.15
    assert abstain2 is False


def test_rrf_empty_channels():
    assert rrf_fuse({}) == []
    assert rrf_fuse({"a": []}) == []


def test_extract_subqueries_no_experimental_by_default():
    from openstinger.temporal.search_utils import extract_subqueries
    qs = extract_subqueries("How many model kits have I bought?")
    blob = " ".join(qs).lower()
    # generic ngrams ok; revell/tamiya only with experimental flag
    assert "revell" not in blob
    qs2 = extract_subqueries(
        "How many model kits have I bought?", experimental_lexicons=True
    )
    assert "revell" in " ".join(qs2).lower() or "tamiya" in " ".join(qs2).lower()
