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


def test_focus_terms_capture_venues_and_acronyms():
    from openstinger.temporal.search_utils import (
        extract_query_focus_terms,
        extract_temporal_anchors,
        is_temporal_span_query,
    )

    q = (
        "How many days passed between my visit to the Museum of Modern Art (MoMA) "
        "and the Ancient Civilizations exhibit at the Metropolitan Museum of Art?"
    )
    assert is_temporal_span_query(q)
    focus = [t.lower() for t in extract_query_focus_terms(q)]
    assert any("moma" == t for t in focus)
    assert any("museum of modern art" == t for t in focus)
    assert any("metropolitan" in t for t in focus)

    q2 = (
        "How many months have passed since I participated in two charity events "
        "in a row, on consecutive days?"
    )
    assert is_temporal_span_query(q2)
    anchors = [t.lower() for t in extract_temporal_anchors(q2)]
    assert "charity" in anchors or "events" in anchors


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
    from openstinger.temporal.search_utils import build_inventory_digest, is_errand_count_query

    assert is_errand_count_query("How many clothing items do I need to pick up?")
    assert not is_errand_count_query("How many model kits have I worked on or bought?")

    hits = [
        {
            "source_description": "s1",
            "cues": {
                "actions": [
                    "returned boots to Zara",
                    "exchanged them for a larger size",
                    "pick up the new pair at Zara",
                ]
            },
            "content": "x",
        },
        {
            "source_description": "s2",
            "cues": {"actions": ["pick up navy blazer from dry cleaning"]},
            "content": "y",
        },
    ]
    dig = build_inventory_digest(hits, "How many clothing items do I need to pick up?")
    # boots return/pickup stay separate from each other and from drycleaning
    assert dig.count("- [") >= 2
    assert "boots" in dig.lower()
    assert "drycleaning" in dig.lower() or "PICKUP" in dig
    assert "EACH tagged line" not in dig
    assert "Distinct items listed:" not in dig
    assert "Suggested outstanding obligations listed:" in dig
    assert "s1" in dig and "s2" in dig

    # Purchase-history counts should not emit an errand digest
    assert build_inventory_digest(hits, "How many model kits have I worked on or bought?") == ""


def test_inventory_digest_skips_policy_noise():
    from openstinger.temporal.search_utils import build_inventory_digest

    hits = [
        {
            "source_description": "noise",
            "cues": {
                "actions": [
                    "return policy for this product?",
                    "exchange rates, and commodity prices",
                ]
            },
            "content": "Best Buy return policy FAQ",
        }
    ]
    assert build_inventory_digest(
        hits, "How many items do I need to return from a store?"
    ) == ""


def test_inventory_digest_skips_advice_and_keeps_real_errands():
    from openstinger.temporal.search_utils import build_inventory_digest

    hits = [
        {
            "source_description": "s1",
            "cues": {
                "actions": [
                    "pick up your new boots and return any items that don't fit quite right",
                    "return some boots to Zara",
                    "pick up the new pair at Zara",
                ]
            },
            "content": "x",
        },
        {
            "source_description": "s2",
            "cues": {
                "actions": [
                    "pick up my dry cleaning for the navy blue blazer",
                ]
            },
            "content": "y",
        },
    ]
    dig = build_inventory_digest(
        hits, "How many items of clothing do I need to pick up or return from a store?"
    )
    assert "don't fit" not in dig.lower()
    assert "Suggested outstanding obligations listed: 3." in dig
    assert "boots" in dig.lower()
    assert "drycleaning" in dig.lower() or "blazer" in dig.lower()


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


def test_query_noun_and_preference_context_boost():
    from openstinger.temporal.search_utils import (
        apply_preference_context_boost,
        apply_query_noun_boost,
        extract_topic_nouns,
        is_preference_context_query,
    )

    assert "kits" in extract_topic_nouns("How many model kits have I worked on or bought?") or "kit" in extract_topic_nouns(
        "How many model kits have I worked on or bought?"
    )
    from openstinger.temporal.search_utils import extract_topic_phrases
    phrases = extract_topic_phrases("How many model kits have I worked on or bought?")
    assert any("kit" in p for p in phrases)
    music_nouns = extract_topic_nouns(
        "How many music albums or EPs have I purchased or downloaded?"
    )
    assert "album" in music_nouns or "albums" in music_nouns
    assert "ep" in music_nouns or "eps" in music_nouns
    assert "vinyl" in music_nouns
    assert is_preference_context_query(
        "I noticed my bike seems to be performing even better during my Sunday group rides. "
        "Could there be a reason for this?"
    )
    rows = [
        {"uuid": "kit", "content": "I bought a Tamiya Spitfire kit last month.", "score": 0.05},
        {"uuid": "noise", "content": "I like pizza on Sundays.", "score": 0.08},
    ]
    boosted = apply_query_noun_boost(rows, "How many model kits have I bought?")
    by_id = {r["uuid"]: r for r in boosted}
    assert by_id["kit"]["score"] > by_id["noise"]["score"]

    bike_rows = [
        {
            "uuid": "maint",
            "content": "I replaced the bike chain and cassette last week.",
            "score": 0.05,
        },
        {"uuid": "other", "content": "Sunday brunch was great.", "score": 0.2},
    ]
    ctx = apply_preference_context_boost(
        bike_rows,
        "I noticed my bike seems to be performing better. Could there be a reason?",
    )
    by_b = {r["uuid"]: r for r in ctx}
    assert by_b["maint"]["score"] > by_b["other"]["score"]
    assert by_b["maint"].get("preference_boost")


def test_extract_event_atoms():
    from openstinger.temporal.search_utils import extract_event_atoms

    text = (
        "user: I visited my dermatologist yesterday for a skin check. "
        "It went well.\n"
        "assistant: Glad to hear! You should keep monitoring it.\n"
        "user: I also started a new yoga class last week. "
        "I'm thinking about buying a new mat someday. "
        "I would go to Paris if I could."
    )
    atoms = extract_event_atoms(text)
    joined = " | ".join(atoms).lower()
    assert "dermatologist" in joined
    assert "yoga" in joined
    # Plans / hypotheticals excluded
    assert "paris" not in joined
    assert "someday" not in joined


def test_coverage_select_and_sibling_harvest():
    from openstinger.temporal.search_utils import (
        coverage_select_for_aggregate,
        harvest_sibling_terms,
        is_aggregate_query,
    )

    q = "How many different doctors did I visit?"
    assert is_aggregate_query(q)
    assert not is_aggregate_query("Where does my sister Emily live?")
    cands = [
        {"uuid": "a1", "source_description": "s1", "score": 0.9,
         "content": "I visited my primary care doctor Dr. Smith on Monday."},
        {"uuid": "a2", "source_description": "s1", "score": 0.85,
         "content": "More chat about the doctor visit and Dr. Smith."},
        {"uuid": "b1", "source_description": "s2", "score": 0.2,
         "content": "The dermatologist doctor, Dr. Jones, checked my skin."},
        {"uuid": "c1", "source_description": "s3", "score": 0.1,
         "content": "Nothing medical here, just cooking pasta."},
    ]
    out = coverage_select_for_aggregate(cands, q, 10)
    srcs = [r["source_description"] for r in out[:2]]
    assert "s1" in srcs and "s2" in srcs

    sib = harvest_sibling_terms(
        [{"content": "I visited a doctor named Dr. Ramirez at Cedars Clinic."}], q
    )
    assert any("ramirez" in t or "cedars" in t for t in sib)


def test_aggregate_reading_digest_surfaces_counts_and_money():
    from openstinger.temporal.search_utils import (
        build_aggregate_reading_digest,
        extract_and_conjuncts,
        is_aggregate_query,
    )

    assert {
        x.lower()
        for x in extract_and_conjuncts(
            "How many plants did I initially plant for tomatoes and cucumbers?"
        )
    } == {"tomatoes", "cucumbers"}
    assert {
        x.lower()
        for x in extract_and_conjuncts(
            "What is the minimum amount I could get if I sold the vintage diamond necklace and the antique vanity?"
        )
    } == {"diamond necklace", "antique vanity"}
    assert {
        x.lower()
        for x in extract_and_conjuncts(
            "How much would I save by taking the bus instead of a taxi?"
        )
    } == {"bus", "taxi"}
    assert {
        x.lower()
        for x in extract_and_conjuncts(
            "What is the total number of goals and assists I have in the recreational indoor soccer league?"
        )
    } == {"goals", "assists"}
    assert {
        x.lower()
        for x in extract_and_conjuncts(
            "What is the total number of lunch meals I got from the chicken fajitas and lentil soup?"
        )
    } == {"chicken fajitas", "lentil soup"}

    assert is_aggregate_query(
        "What is the minimum amount I could get if I sold the necklace and vanity?"
    )
    hits = [
        {
            "source_description": "s1",
            "content": "User: I planted 5 tomato plants in the backyard.",
            "valid_at_human": "2023-03-01",
        },
        {
            "source_description": "s2",
            "content": "User: I also planted 3 cucumber plants near the fence.",
            "valid_at_human": "2023-03-08",
        },
        {
            "source_description": "s3",
            "content": "User: I'm selling my vintage diamond necklace, which is worth $5,000.",
            "valid_at_human": "2023-04-01",
        },
        {
            "source_description": "s4",
            "content": "User: The antique vanity is worth at least $150 after I restored it.",
            "valid_at_human": "2023-04-02",
        },
    ]
    plants = build_aggregate_reading_digest(
        hits, "How many plants did I initially plant for tomatoes and cucumbers?"
    )
    assert "Suggested aggregate count" in plants
    assert "8" in plants
    money = build_aggregate_reading_digest(
        hits,
        "What is the minimum amount I could get if I sold the vintage diamond necklace and the antique vanity?",
    )
    assert "Suggested money total" in money
    assert "5150" in money
    # Incomplete conjunct: only one side priced → do not invent a total
    incomplete = build_aggregate_reading_digest(
        hits[:3],
        "What is the minimum amount I could get if I sold the vintage diamond necklace and the antique vanity?",
    )
    assert "Incomplete money evidence" in incomplete
    # Open enumeration must not force a blind sum of every integer
    films = build_aggregate_reading_digest(
        [
            {
                "source_description": "f1",
                "content": "User: I watched Iron Man and also saw 2 trailers.",
                "valid_at_human": "a",
            },
            {
                "source_description": "f2",
                "content": "User: I watched Thor last month.",
                "valid_at_human": "b",
            },
        ],
        "How many MCU films did I watch in the last 3 months?",
    )
    assert "Suggested aggregate count (sum" not in films
    assert "Enumerate" in films or "distinct" in films.lower() or "Candidate" in films


def test_temporal_span_excludes_content_and_spend_lookups():
    from openstinger.temporal.search_utils import (
        asks_temporal_span_integer,
        is_temporal_ago_query,
        is_temporal_span_query,
    )

    # Duration counts: keep
    assert is_temporal_ago_query("How many weeks ago did I start using Ibotta?")
    assert asks_temporal_span_integer(
        "How many days passed between my visit to MoMA and the Met?"
    )
    assert asks_temporal_span_integer(
        "How many weeks have I been taking sculpting classes when I bought tools?"
    )
    assert asks_temporal_span_integer(
        "How many days did it take for my shutter release cable to arrive after I ordered it?"
    )
    # Watch/read effort totals are NOT calendar spans
    assert not asks_temporal_span_integer(
        "How many weeks did it take me to watch all the Marvel Cinematic Universe "
        "movies and the main Star Wars films?"
    )
    from openstinger.temporal.search_utils import is_stated_effort_duration_query

    assert is_stated_effort_duration_query(
        "How many weeks did it take me to watch all the Marvel Cinematic Universe "
        "movies and the main Star Wars films?"
    )

    # Content / order / spend: must not force a bare integer
    content_qs = [
        "Which book did I finish a week ago?",
        "I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?",
        "What was the significant business milestone I mentioned four weeks ago?",
        "What is the order of the three events: ShopRite, Walmart, and Ibotta?",
        "How much total money did I spend on attending workshops in the last four months?",
        "How many days did I spend in total traveling in Hawaii and in New York City?",
        "How many years older am I than when I graduated from college?",
    ]
    for q in content_qs:
        assert not is_temporal_ago_query(q), q
        assert not is_temporal_span_query(q), q
        assert not asks_temporal_span_integer(q), q


def test_activity_sum_and_temporal_ago_digest():
    from openstinger.temporal.search_utils import (
        build_activity_duration_digest,
        build_temporal_span_digest,
        is_temporal_ago_query,
    )

    assert is_temporal_ago_query("How many weeks ago did I start using Ibotta?")
    hits = [
        {
            "source_description": "g1",
            "content": "I spent around 70 hours playing Assassin's Creed Odyssey.",
            "valid_at_human": "2023-05-01",
        },
        {
            "source_description": "g2",
            "content": "It took me 30 hours to finish The Last of Us Part II on hard.",
            "valid_at_human": "2023-05-10",
        },
        {
            "source_description": "g3",
            "content": "Celeste took me 10 hours to complete.",
            "valid_at_human": "2023-05-12",
        },
        {
            "source_description": "g4",
            "content": "It took me 25 hours to complete on normal difficulty.",
            "valid_at_human": "2023-05-15",
        },
        {
            "source_description": "g5",
            "content": "Hyper Light Drifter, which took me 5 hours to finish.",
            "valid_at_human": "2023-05-18",
        },
    ]
    ad = build_activity_duration_digest(
        hits, "How many hours have I spent playing games in total?"
    )
    assert "Suggested total hours listed: 140" in ad
    assert "Candidate duration sum" in ad

    span_hits = [
        {
            "source_description": "b1",
            "content": "I attended a baking class at a local culinary school.",
            "valid_at_human": "2022-03-25",
        }
    ]
    sd = build_temporal_span_digest(
        span_hits,
        "How many days ago did I attend a baking class at a local culinary school?",
    )
    assert "ago" in sd.lower()
    assert "question_date" in sd or "baking" in sd.lower()


def test_rewatch_count_prefers_titles_not_watched_n():
    from openstinger.temporal.search_utils import (
        build_aggregate_reading_digest,
        is_rewatch_count_query,
    )

    q = "How many Marvel movies did I re-watch?"
    assert is_rewatch_count_query(q)
    hits = [
        {
            "source_description": "r1",
            "content": (
                "user: Since I just re-watched Avengers: Endgame yesterday, "
                "I've been thinking about other movies. I've actually watched "
                "Doctor Strange already, it was one of the four Marvel movies "
                "I watched recently."
            ),
        },
        {
            "source_description": "r2",
            "content": (
                "user: I've been into Marvel movies lately, I also re-watched "
                "Spider-Man: No Way Home, which is another Marvel movie."
            ),
        },
    ]
    d = build_aggregate_reading_digest(hits, q)
    assert "Suggested distinct item count: 2" in d
    assert "Suggested stated total from first-person count claim:" not in d
    assert "Avengers: Endgame" in d
    assert "Spider-Man: No Way Home" in d


def test_aggregate_enumerate_acquire_and_rewatch_verbs():
    from openstinger.temporal.search_utils import build_aggregate_reading_digest

    # Gold shape: two acquires in one utterance + one "which I got from"
    along_hits = [
        {
            "source_description": "a1",
            "content": (
                "user: I'm caring for my peace lily, which I got from the nursery "
                "two weeks ago along with a succulent."
            ),
        },
        {
            "source_description": "a2",
            "content": (
                "user: My snake plant, which I got from my sister last month, "
                "is doing great."
            ),
        },
    ]
    ad = build_aggregate_reading_digest(
        along_hits, "How many plants did I acquire in the last month?"
    )
    assert "Suggested distinct item count: 3" in ad
    assert "along with" in ad.lower() or "succulent" in ad.lower()
    # "two weeks ago" must not become Candidate numeric claim 2
    assert "Candidate numeric claims: 2←" not in ad

    movie_hits = [
        {
            "source_description": "m1",
            "content": "user: I re-watched the Marvel movie Iron Man last night.",
        },
        {
            "source_description": "m2",
            "content": "user: I rewatched another Marvel movie, The Avengers.",
        },
    ]
    md = build_aggregate_reading_digest(
        movie_hits, "How many Marvel movies did I re-watch?"
    )
    assert "Suggested distinct item count: 2" in md


def test_days_before_span_pairs_later_and_earlier():
    from openstinger.temporal.search_utils import (
        asks_temporal_span_integer,
        build_temporal_span_digest,
    )

    q = (
        "How many days before I bought the iPhone 13 Pro did I attend "
        "the Holiday Market?"
    )
    assert asks_temporal_span_integer(q)
    hits = [
        {
            "source_description": "m1",
            "content": "I went to the Holiday Market downtown with friends.",
            "valid_at_human": "2023/12/10",
        },
        {
            "source_description": "m2",
            "content": "I finally bought the iPhone 13 Pro today.",
            "valid_at_human": "2023/12/17",
        },
        {
            "source_description": "noise",
            "content": "I bought coffee and attended a webinar.",
            "valid_at_human": "2023/11/01",
        },
    ]
    digest = build_temporal_span_digest(hits, q)
    assert "Suggested span from listed event dates: 7 days" in digest
    assert "Answer with the integer 7" in digest

    same_stamp = [
        {
            "source_description": "s1",
            "content": "I ordered her gift online for the birthday party.",
            "valid_at_human": "2023/06/01",
        },
        {
            "source_description": "s1b",
            "content": "My best friend's birthday party was wonderful.",
            "valid_at_human": "2023/06/01",
        },
    ]
    # Collapsed stamps must not force Suggested span 0 into the answer path.
    zero_q = (
        "How many days before my best friend's birthday party did I order her gift?"
    )
    zero_digest = build_temporal_span_digest(same_stamp, zero_q)
    assert "Suggested span from listed event dates: 0" not in zero_digest


def test_soft_advice_and_temporal_ago_detection():
    from openstinger.temporal.search_utils import (
        apply_preference_boost,
        is_activity_duration_query,
        is_soft_advice_query,
        is_temporal_span_query,
        needs_preference_retrieval,
    )

    tips = "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?"
    dinner = "What should I serve for dinner this weekend with my homegrown ingredients?"
    advice = "I've been struggling with my slow cooker recipes. Any advice on getting better results?"
    from openstinger.temporal.search_utils import soft_advice_bridge_terms

    assert is_soft_advice_query(tips)
    assert is_soft_advice_query(dinner)
    assert is_soft_advice_query(advice)
    assert is_soft_advice_query(
        "I was thinking of trying a new coffee creamer recipe. Any recommendations?"
    )
    assert is_soft_advice_query("Can you suggest some useful accessories for my phone?")
    assert needs_preference_retrieval(tips)
    assert needs_preference_retrieval(dinner)
    assert not is_soft_advice_query("Can you recommend a hotel in Miami?")
    kit_bridges = soft_advice_bridge_terms(tips)
    assert "utensil" in kit_bridges or "utensil holder" in kit_bridges
    dinner_bridges = soft_advice_bridge_terms(dinner)
    assert "garden" in dinner_bridges or "tomato" in dinner_bridges
    phone = "I've been having trouble with the battery life on my phone lately. Any tips?"
    assert "power bank" in soft_advice_bridge_terms(phone)
    guitar = (
        "I'm getting excited about my visit to the music store this weekend. "
        "Any tips on what to look for in a new guitar?"
    )
    assert "guitar" in soft_advice_bridge_terms(guitar)
    creamer = "I was thinking of trying a new coffee creamer recipe. Any recommendations?"
    creamer_bridges = soft_advice_bridge_terms(creamer)
    assert "creamer" in creamer_bridges or "almond milk" in creamer_bridges
    assert "tomato" not in creamer_bridges and "garden" not in creamer_bridges
    assert is_temporal_span_query(
        "How many days ago did I attend a baking class at a local culinary school?"
    )
    assert is_activity_duration_query(
        "How many hours did I spend playing games last week?"
    )
    rows = [
        {
            "uuid": "kit",
            "content": "I love my new utensil holder and worry about the granite near the sink.",
            "score": 0.05,
            "cues": {"preferences": ["keep granite clean"], "features": ["utensil holder"]},
        },
        {"uuid": "noise", "content": "I watched a movie last night.", "score": 0.2},
    ]
    boosted = apply_preference_boost(rows, tips)
    by_id = {r["uuid"]: r for r in boosted}
    assert by_id["kit"]["score"] >= by_id["noise"]["score"]


def test_activity_and_topic_digests():
    from openstinger.temporal.search_utils import (
        build_activity_duration_digest,
        build_topic_inventory_digest,
        is_activity_duration_query,
        is_topic_inventory_query,
    )

    assert is_activity_duration_query("How many hours of jogging and yoga did I do last week?")
    assert is_topic_inventory_query("How many model kits have I worked on or bought?")
    assert is_topic_inventory_query(
        "How many music albums or EPs have I purchased or downloaded?"
    )
    assert not is_topic_inventory_query(
        "How many days passed between the day I cancelled my subscription "
        "and the day I did my online grocery shopping?"
    )
    act_hits = [
        {
            "source_description": "s1",
            "content": "I did 30 minutes of yoga on Tuesday and a short jog.",
        }
    ]
    ad = build_activity_duration_digest(
        act_hits, "How many hours of jogging and yoga did I do last week?"
    )
    assert "30 minutes" in ad.lower() or "minutes" in ad.lower()

    kit_hits = [
        {"source_description": "k1", "content": "Started the Revell F-15 Eagle model kit."},
        {"source_description": "k2", "content": "Bought a Tamiya Spitfire kit at 1/48 scale."},
    ]
    td = build_topic_inventory_digest(
        kit_hits, "How many model kits have I worked on or bought?"
    )
    assert (
        "Sessions with topic overlap listed:" in td
        or "Suggested distinct purchases/downloads listed:" in td
    )
    assert "k1" in td and "k2" in td
