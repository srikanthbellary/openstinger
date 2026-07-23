"""Temporal span side-pairing and order digests (fair_c4 fail cluster)."""

from __future__ import annotations

from openstinger.temporal.search_utils import (
    build_temporal_order_digest,
    build_temporal_span_digest,
    is_shipping_latency_query,
    is_temporal_order_query,
    temporal_span_side_bridge_terms,
)


def test_since_when_ukulele_span():
    q = (
        "How many days had passed since I started taking ukulele lessons "
        "when I decided to take my acoustic guitar to the guitar tech for servicing?"
    )
    hits = [
        {
            "source_description": "u1",
            "valid_at": 1675234680,  # 2023-02-01
            "valid_at_human": "2023/02/01",
            "content": (
                "user: I just started taking ukulele lessons with my friend Rachel today."
            ),
        },
        {
            "source_description": "u2",
            "valid_at": 1677328440,  # 2023-02-25
            "valid_at_human": "2023/02/25",
            "content": (
                "user: I decided to take my Taylor GS Mini to the guitar tech "
                "for servicing today."
            ),
        },
        {
            "source_description": "noise",
            "valid_at": 1673481600,  # 2023-01-12
            "valid_at_human": "2023/01/12",
            "content": "user: I bought a camera lens today.",
        },
    ]
    assert "ukulele" in temporal_span_side_bridge_terms(q)
    d = build_temporal_span_digest(hits, q, question_date="2023/04/01")
    assert "Suggested span from listed event dates: 24 days" in d


def test_between_day_spark_plugs():
    q = (
        "How many days passed between the day I replaced my spark plugs and "
        "the day I participated in the Turbocharged Tuesdays auto racking event?"
    )
    hits = [
        {
            "source_description": "s1",
            "valid_at": 1676392140,
            "valid_at_human": "2023/02/14",
            "content": (
                "user: I replaced my spark plugs with new ones from NGK today."
            ),
        },
        {
            "source_description": "s2",
            "valid_at": 1678876800,
            "valid_at_human": "2023/03/15",
            "content": (
                "user: I participated in the Turbocharged Tuesdays auto "
                "racking event today."
            ),
        },
    ]
    d = build_temporal_span_digest(hits, q, question_date="2023/04/01")
    assert "Suggested span from listed event dates: 29 days" in d


def test_shipping_weeks_buy_receive():
    q = (
        "How many weeks passed between the day I bought my new tennis racket "
        "and the day I received it?"
    )
    assert is_shipping_latency_query(q)
    hits = [
        {
            "source_description": "t1",
            "valid_at": 1678449780,
            "valid_at_human": "2023/03/10",
            "content": (
                "user: I just bought a new tennis racket online today."
            ),
        },
        {
            "source_description": "t2",
            "valid_at": 1679044440,
            "valid_at_human": "2023/03/17",
            "content": (
                "user: I just received my new tennis racket today and I'm excited."
            ),
        },
    ]
    d = build_temporal_span_digest(hits, q, question_date="2023/04/15")
    assert "Suggested span from listed event dates: 1 weeks" in d or (
        "Suggested span from listed event dates: 1 week" in d
    )


def test_order_query_not_integer():
    q = (
        "What is the order of the three sports events I participated in "
        "during the past month, from earliest to latest?"
    )
    assert is_temporal_order_query(q)
    hits = [
        {
            "source_description": "e1",
            "valid_at": 1685721600,
            "valid_at_human": "2023/06/02",
            "content": (
                "user: I just completed the Spring Sprint Triathlon today."
            ),
        },
        {
            "source_description": "e2",
            "valid_at": 1686412800,
            "valid_at_human": "2023/06/10",
            "content": (
                "user: I just finished a 5K run at the Midsummer 5K Run today."
            ),
        },
        {
            "source_description": "e3",
            "valid_at": 1687017600,
            "valid_at_human": "2023/06/17",
            "content": (
                "user: I participate in the company's annual charity soccer "
                "tournament today."
            ),
        },
    ]
    d = build_temporal_order_digest(hits, q)
    assert "Chronological event order" in d
    assert "Spring Sprint Triathlon" in d
    assert "Midsummer 5K" in d
    assert "soccer" in d.lower()
    assert "integer count" in d.lower()
