"""Unit tests for fair_c4b fix digests (structural, no QID branches)."""

from openstinger.temporal.search_utils import (
    build_age_delta_digest,
    build_aggregate_reading_digest,
    build_pairwise_first_digest,
    build_role_tenure_digest,
    build_tank_population_digest,
    build_types_of_digest,
    is_age_delta_query,
    is_comparative_savings_query,
    is_pairwise_first_query,
    is_role_tenure_query,
    is_tank_population_query,
    is_types_of_count_query,
)


def test_tank_population_sums_across_aquariums():
    q = "How many fish are there in total in both of my aquariums?"
    assert is_tank_population_query(q)
    hits = [
        {
            "source_description": "t1",
            "content": (
                "user: my new 20-gallon tank, which currently has 10 neon tetras, "
                "5 golden honey gouramis, and a small pleco catfish."
            ),
        },
        {
            "source_description": "t2",
            "content": (
                "user: I also upgraded my old 10-gallon tank, which has my "
                "betta fish, Bubbles."
            ),
        },
    ]
    d = build_tank_population_digest(hits, q)
    assert "Suggested stated total from first-person count claim: 17." in d


def test_comparative_savings_difference():
    q = (
        "How much will I save by taking the train from the airport to my hotel "
        "instead of a taxi?"
    )
    assert is_comparative_savings_query(q)
    hits = [
        {
            "source_description": "m1",
            "content": (
                "user: taking a taxi from the airport to my hotel would cost "
                "around $60, which is a bit pricey for me."
            ),
        },
        {
            "source_description": "m2",
            "content": (
                "user: it's actually $10 to get to my hotel from the airport "
                "by train."
            ),
        },
    ]
    d = build_aggregate_reading_digest(hits, q)
    assert "Suggested savings difference: 50" in d


def test_comments_metric_conjunct_sum():
    q = (
        "What is the total number of comments on my recent Facebook Live "
        "session and my most popular YouTube video?"
    )
    hits = [
        {
            "source_description": "c1",
            "content": (
                "user: my recent Facebook Live session that got 12 comments."
            ),
        },
        {
            "source_description": "c2",
            "content": (
                "user: My most popular YouTube video has 21 comments, and I "
                "wish to do better than that."
            ),
        },
    ]
    d = build_aggregate_reading_digest(hits, q)
    assert "Suggested aggregate count (sum of per-conjunct claims): 33" in d


def test_pairwise_first_uses_relative_ago():
    q = (
        "Which device did I set up first, the smart thermostat or the "
        "mesh network system?"
    )
    assert is_pairwise_first_query(q)
    hits = [
        {
            "source_description": "d1",
            "valid_at_human": "2023/05/25",
            "content": (
                "user: I recently upgraded my home Wi-Fi router to a new "
                "mesh network system."
            ),
        },
        {
            "source_description": "d2",
            "valid_at_human": "2023/05/25",
            "content": (
                "user: since I set up my smart thermostat a month ago, I've "
                "noticed that my bills are lower."
            ),
        },
    ]
    d = build_pairwise_first_digest(hits, q)
    assert "Suggested first event: smart thermostat" in d


def test_pairwise_first_uses_session_stamps():
    from openstinger.temporal.search_utils import build_pairwise_first_digest, is_pairwise_first_query

    q = (
        "Which event happened first, my cousin's wedding or Michael's "
        "engagement party?"
    )
    assert is_pairwise_first_query(q)
    hits = [
        {
            "source_description": "w1",
            "valid_at": 1_700_000_000,
            "valid_at_human": "2023/11/14",
            "content": "user: I went to my cousin's wedding last weekend.",
        },
        {
            "source_description": "e1",
            "valid_at": 1_680_000_000,
            "valid_at_human": "2023/03/28",
            "content": (
                "user: I just came back from Michael's engagement party recently."
            ),
        },
    ]
    d = build_pairwise_first_digest(hits, q)
    assert "Suggested first event: Michael's engagement party" in d


def test_pairwise_first_last_month_beats_weeks_ago():
    from openstinger.temporal.search_utils import build_pairwise_first_digest

    q = (
        "Which event happened first, the purchase of the coffee maker or the "
        "malfunction of the stand mixer?"
    )
    hits = [
        {
            "source_description": "c1",
            "valid_at_human": "2023/05/25",
            "content": (
                "user: I bought my coffee maker about three weeks ago and it "
                "has been acting up."
            ),
        },
        {
            "source_description": "m1",
            "valid_at_human": "2023/05/25",
            "content": (
                "user: I had to take my stand mixer to a repair shop last month "
                "after a malfunction."
            ),
        },
    ]
    d = build_pairwise_first_digest(hits, q)
    assert "Suggested first event: malfunction of the stand mixer" in d


def test_pairwise_first_strips_quotes():
    from openstinger.temporal.search_utils import build_pairwise_first_digest

    q = (
        "Which event did I attend first, the 'Effective Time Management' "
        "workshop or the 'Data Analysis using Python' webinar?"
    )
    hits = [
        {
            "source_description": "w1",
            "valid_at_human": "2023/05/24",
            "content": (
                'user: I attended the workshop on "Effective Time Management" '
                "last Saturday."
            ),
        },
        {
            "source_description": "d1",
            "valid_at_human": "2023/05/24",
            "content": (
                'user: I participated in a webinar on "Data Analysis using '
                'Python" two months ago.'
            ),
        },
    ]
    d = build_pairwise_first_digest(hits, q)
    assert "Suggested first event: Data Analysis using Python webinar" in d


def test_personal_best_prefers_newest_claim():
    from openstinger.temporal.search_utils import build_personal_best_digest

    q = "What was my personal best time in the charity 5K run?"
    hits = [
        {
            "source_description": "old",
            "valid_at_human": "2023/05/23",
            "content": (
                "user: I recently set a personal best time in a charity 5K "
                "run with a time of 27:12."
            ),
        },
        {
            "source_description": "new",
            "valid_at_human": "2023/05/30",
            "content": (
                "user: I'm hoping to beat my personal best time of 25:50 "
                "this time around."
            ),
        },
    ]
    d = build_personal_best_digest(hits, q)
    assert "Suggested current personal best: 25:50" in d


def test_friends_and_family_not_conjuncts():
    from openstinger.temporal.search_utils import extract_and_conjuncts

    q = (
        "How many babies were born to friends and family members in the "
        "last few months?"
    )
    assert extract_and_conjuncts(q) == []


def test_pairwise_incomplete_pair_abstains():
    from openstinger.temporal.search_utils import build_pairwise_first_digest

    q = "Who became a parent first, Tom or Alex?"
    hits = [
        {
            "source_description": "a1",
            "valid_at_human": "2023/01/15",
            "content": "user: Alex became a parent in January when they adopted.",
        },
    ]
    d = build_pairwise_first_digest(hits, q)
    assert "Suggested answer: I do not know" in d
    assert "Tom" in d


def test_weekday_wake_offset():
    from openstinger.temporal.search_utils import (
        build_weekday_wake_digest,
        is_weekday_wake_query,
    )

    q = "What time do I wake up on Tuesdays and Thursdays?"
    assert is_weekday_wake_query(q)
    hits = [
        {
            "source_description": "w1",
            "content": (
                "user: I usually wake up at 7:00 AM. On Tuesdays and Thursdays "
                "I wake up 15 minutes earlier for yoga."
            ),
        },
    ]
    d = build_weekday_wake_digest(hits, q)
    assert "Suggested wake time: 6:45 AM" in d


def test_content_ago_pins_near_target():
    from openstinger.temporal.search_utils import (
        build_content_ago_digest,
        is_content_ago_query,
    )

    q = "I mentioned an investment for a competition four weeks ago? What did I buy?"
    assert is_content_ago_query(q)
    hits = [
        {
            "source_description": "later",
            "valid_at_human": "2023/06/20",
            "content": (
                "user: I bought a modeling tool set, a wire cutter, and a "
                "sculpting mat yesterday."
            ),
        },
        {
            "source_description": "earlier",
            "valid_at_human": "2023/05/20",
            "content": (
                "user: I got my own set of sculpting tools for the competition."
            ),
        },
    ]
    d = build_content_ago_digest(hits, q, question_date="2023-06-17")
    assert "Content-ago time pin" in d
    assert "earlier" in d
    assert "Nearest candidate session: earlier" in d


def test_ago_span_never_suggests_zero():
    from openstinger.temporal.search_utils import build_temporal_span_digest

    q = "How many months ago did I book the Airbnb in San Francisco?"
    hits = [
        {
            "source_description": "a1",
            "valid_at_human": "2023/06/17",
            "content": "user: I booked an Airbnb in San Francisco today.",
        },
    ]
    d = build_temporal_span_digest(hits, q, question_date="2023-06-17")
    assert "Suggested span from listed event dates: 0" not in d
    assert "Do not answer 0" in d


def test_ku_previous_selects_second_newest():
    from openstinger.temporal.search_utils import prioritize_query_entity_recency

    q = "What was my previous personal best time in the charity 5K run?"
    eps = [
        {
            "uuid": "new",
            "valid_at": 300,
            "content": "user: my personal best is now 25:50 in the charity 5K.",
        },
        {
            "uuid": "old",
            "valid_at": 200,
            "content": "user: my personal best in the charity 5K was 27:45.",
        },
        {
            "uuid": "mid",
            "valid_at": 250,
            "content": "user: I ran a 27:12 charity 5K today.",
        },
    ]
    out = prioritize_query_entity_recency(eps, q)
    assert out[0]["uuid"] == "old"
    assert "PREVIOUS value" in (out[0].get("content") or "")


def test_role_tenure_difference():
    q = "How long have I been working in my current role?"
    assert is_role_tenure_query(q)
    hits = [
        {
            "source_description": "r1",
            "content": (
                "user: started as a Marketing Coordinator and worked my way up "
                "to Senior Marketing Specialist after 2 years and 4 months."
            ),
        },
        {
            "source_description": "r2",
            "content": (
                "user: thinking about my 3 years and 9 months experience in "
                "the company."
            ),
        },
    ]
    d = build_role_tenure_digest(hits, q)
    assert "Suggested role tenure: 1 year and 5 months" in d


def test_age_delta_current_minus_grad_age():
    q = "How many years older am I than when I graduated from college?"
    assert is_age_delta_query(q)
    hits = [
        {
            "source_description": "a1",
            "content": (
                "user: I completed my degree at the age of 25 from Berkeley."
            ),
        },
        {
            "source_description": "a2",
            "content": (
                "user: As a 32-year-old Digital Marketing Specialist, I want "
                "to upskill."
            ),
        },
    ]
    d = build_age_delta_digest(hits, q)
    assert "Suggested stated total from first-person count claim: 7." in d


def test_types_of_citrus():
    q = (
        "How many different types of citrus fruits have I used in my "
        "cocktail recipes?"
    )
    assert is_types_of_count_query(q)
    hits = [
        {
            "source_description": "a",
            "content": "user: made my own orange bitters using orange peels.",
        },
        {
            "source_description": "b",
            "content": "user: summer drinks that use fresh lime juice.",
        },
        {
            "source_description": "c",
            "content": (
                "user: Sangria in a pitcher with slices of orange and lemon."
            ),
        },
    ]
    d = build_types_of_digest(hits, q)
    assert "Suggested stated total from first-person count claim: 3." in d
