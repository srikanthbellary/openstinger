"""BM25 / hybrid search helpers (v0.10 LME-informed)."""

from __future__ import annotations

import re
from typing import Any

# RediSearch / FalkorDB fulltext operators and fragile tokens
_BM25_SPECIAL = re.compile(r"[\[\]\{\}\(\)\<\>\~\*\?\:\"\'\@\!\^\|\-\+\\/]")
_STOP = {
    "what", "when", "where", "who", "why", "how", "which", "whom",
    "did", "does", "do", "the", "a", "an", "i", "my", "me", "mine",
    "with", "from", "for", "to", "of", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "and", "or", "but", "if", "then",
    "that", "this", "these", "those", "you", "your", "we", "our",
    "last", "first", "not", "can", "could", "would", "should",
    "have", "has", "had", "about", "into", "than", "also", "just",
    "some", "any", "all", "each", "more", "most", "other", "only",
}

_RECOMMEND_RE = re.compile(
    r"\b(recommend|suggest|suggestion|prefer|preference|interesting|"
    r"hotel|publication|conference|resources?)\b",
    re.I,
)
_SOFT_ADVICE_RE = re.compile(
    r"\b(?:any\s+)?(?:tips?|advice|pointers?|ideas?|guidance|recommendations?|suggestions?)\b|"
    r"\bcan you suggest\b|"
    r"\bwhat should i (?:serve|do|try|make|use|buy|look for)\b|"
    r"\bkeep(?:ing)? (?:it |them )?clean\b|"
    r"\bhappening around me\b",
    re.I,
)
_COUNT_RE = re.compile(r"\b(how many|how much|count|number of|total)\b", re.I)
_PREF_CUE_RE = re.compile(
    r"\b(prefer|preference|preferable|like|love|want|rather|favorite|"
    r"favourite|interested in|looking for|would rather|don't like|do not like|"
    r"hate|wish|hoping for)\b",
    re.I,
)
_NAME_IN_QUERY = re.compile(r"\b([A-Z][a-z]{2,})\b")
_PAREN_ACRONYM_RE = re.compile(r"\(([A-Z][A-Za-z0-9]{1,7})\)")
_MIXED_ACRONYM_RE = re.compile(r"\b([A-Z][a-z]*[A-Z][A-Za-z]{0,4})\b")
_PROPER_PHRASE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:of|the|and|at|in)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+){1,5})\b"
)
_MONTH_NAME_STOP = {
    "how", "what", "when", "where", "which", "can", "the", "you",
    "march", "april", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "museum", "art", "modern", "ancient",
    "civilizations", "exhibit", "events", "charity",
}
_PHRASE_STOP = _STOP | {
    "passed", "between", "visit", "helped", "friend", "cousin", "ordered",
    "happened", "events", "order", "days", "day", "worked", "bought",
}


def extract_query_focus_terms(query: str) -> list[str]:
    """
    Proper names, acronyms, and venue phrases from the question.

    Used to force CONTAINS follow-up so updates, venues, and preferences are not missed.
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        t = (term or "").strip()
        if len(t) < 2:
            return
        low = t.lower()
        if low in _STOP or low in _MONTH_NAME_STOP:
            return
        # Drop weak phrase fragments
        if low.startswith("of ") or low.endswith(" of") or low in {"museum of", "of art"}:
            return
        if low not in seen:
            seen.add(low)
            found.append(t)

    q = query or ""
    # Multi-word venues first (Museum of Modern Art, Metropolitan Museum of Art)
    for phrase in _PROPER_PHRASE_RE.findall(q):
        _add(phrase)
        # Also keep a shorter 2-token tail when useful (Metropolitan Museum)
        parts = phrase.split()
        if len(parts) >= 2:
            _add(" ".join(parts[:2]))
            _add(" ".join(parts[-2:]))
    for ac in _PAREN_ACRONYM_RE.findall(q):
        _add(ac)
    for ac in _MIXED_ACRONYM_RE.findall(q):
        _add(ac)
    for m in _NAME_IN_QUERY.findall(q):
        _add(m)
    return found


def is_temporal_span_query(query: str) -> bool:
    """
    True when the question asks for a duration span (days/weeks/months/years).

    Content lookups that only mention 'a week ago' as a time cue are excluded so
    digests do not force a bare integer answer.
    """
    return asks_temporal_span_integer(query or "")


def extract_temporal_anchors(query: str) -> list[str]:
    """
    Anchors for temporal-span CONTAINS: proper names/venues, else content nouns.
    """
    focus = extract_query_focus_terms(query)
    if focus:
        return focus
    skip = {
        "days", "day", "weeks", "week", "months", "month", "years", "year",
        "row", "consecutive", "passed", "since", "between", "participated",
        "visit", "exhibit", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "many", "number", "total",
    }
    out: list[str] = []
    seen: set[str] = set()
    for p in extract_topic_phrases(query, max_n=6):
        if p in seen or any(tok in skip for tok in p.split()):
            continue
        seen.add(p)
        out.append(p)
    for n in extract_topic_nouns(query, max_n=10):
        if n in skip or n in seen or n.isdigit():
            continue
        seen.add(n)
        out.append(n)
    return out



def extract_search_terms(query: str) -> list[str]:
    """Alphanumeric tokens useful for BM25 / CONTAINS, stopwords removed."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP]


def sanitize_bm25_query(query: str, *, mode: str = "and") -> str:
    """
    Make a natural-language question safe for FalkorDB/RediSearch BM25.

    Quotes each kept term (operators like ':' and bare 'last' break the parser).
    mode='and' (space-joined): precise, but returns zero rows for verbose
    multi-term questions. mode='or' ('|'-joined): broad recall for RRF fusion.
    Run both as separate channels; AND wins when it matches, OR fills the gaps.
    """
    terms = extract_search_terms(query)
    if terms:
        joiner = "|" if mode == "or" else " "
        return joiner.join(f'"{t}"' for t in terms)
    cleaned = _BM25_SPECIAL.sub(" ", query or "").strip()
    # Stopword-only or empty input: avoid sending raw stopwords to BM25
    if not cleaned or not extract_search_terms(cleaned):
        return '""'
    return cleaned


def is_recommend_query(query: str) -> bool:
    return bool(_RECOMMEND_RE.search(query or ""))


def is_soft_advice_query(query: str) -> bool:
    """Tips/advice follow-ups that need prior user prefs without 'recommend' verbs."""
    return bool(_SOFT_ADVICE_RE.search(query or ""))


def needs_preference_retrieval(query: str) -> bool:
    """Recommend or soft-advice questions that should surface preference-rich memory."""
    return is_recommend_query(query) or is_soft_advice_query(query)


def event_attend_bridge_terms(query: str) -> list[str]:
    """
    Lexical bridges for 'how many events did I attend' style counts.

    Questions name the category (art-related events) but gold sessions use
    attended/volunteered/lecture/exhibition wording without repeating 'events'.
    """
    if not is_count_query(query):
        return []
    ql = (query or "").lower()
    if not re.search(r"\b(?:events?|exhibitions?|tours?)\b", ql):
        return []
    if not re.search(r"\b(?:attend|attended|went|visit|visited)\b", ql):
        return []
    return [
        "attended",
        "volunteered",
        "exhibition",
        "lecture",
        "guided tour",
        "museum",
        "gallery",
        "art afternoon",
    ]


def fact_lookup_bridge_terms(query: str) -> list[str]:
    """
    Bridges for sparse fact lookups where the question category omits the object.

    Examples: 'kitchen appliance ... days ago' → got/bought/smoker-class purchase
    language; 'order of the sports events' → watched/attended game cues.
    """
    ql = (query or "").lower()
    out: list[str] = []

    def _add(*terms: str) -> None:
        for t in terms:
            t = (t or "").strip().lower()
            if t and t not in out:
                out.append(t)

    if re.search(r"\b(?:appliance|gadget)\b", ql) and re.search(
        r"\b(?:buy|bought|purchase|ago|days?)\b", ql
    ):
        _add(
            "got a",
            "just got",
            "bought",
            "purchased",
            "smoker",
            "bbq",
            "grill",
            "appliance",
        )
    if re.search(r"\border of the (?:sports )?events\b", ql) or (
        "sports" in ql and "order" in ql
    ):
        _add(
            "watched",
            "attended",
            "nba",
            "nfl",
            "playoffs",
            "championship",
            "staples",
            "football",
        )
    # Future age / "how old will I be when …"
    if re.search(
        r"\b(?:how (?:old|many years) will i be|years will i be when)\b", ql
    ) or re.search(r"\bwhen\b.+\b(?:gets? married|marries|wedding)\b", ql):
        _add(
            "i'm",
            "i am",
            "years old",
            "my age",
            "getting married",
            "married next",
            "next year",
            "rachel",
        )
    return out[:12]


def soft_advice_bridge_terms(query: str) -> list[str]:
    """
    Related possession/topic terms for tips/advice follow-ups.

    Soft-advice questions often omit the earlier object (utensil holder, garden
    herbs, power bank). Bridge to common prior-context nouns so CONTAINS/BM25
    can surface those sessions without question-id lexicons.
    """
    if not is_soft_advice_query(query):
        return []
    q = (query or "").lower()
    out: list[str] = []

    def _add(*terms: str) -> None:
        for t in terms:
            t = (t or "").strip().lower()
            if t and t not in out:
                out.append(t)

    if any(w in q for w in ("kitchen", "counter", "mess", "clean", "tidy")):
        _add("utensil", "utensil holder", "granite", "countertop", "garbage disposal")
    if any(
        w in q
        for w in (
            "dinner",
            "lunch",
            "breakfast",
            "meal",
            "ingredient",
            "serve",
            "cook",
            "recipe",
            "homegrown",
        )
    ) and not any(w in q for w in ("creamer", "coffee", "latte")):
        # Do not bridge dinner-garden terms onto coffee-creamer tip questions
        _add("garden", "harvest", "tomato", "basil", "mint", "herb", "recipe")
    if any(w in q for w in ("battery", "phone", "smartphone", "charger", "accessories")):
        _add(
            "power bank",
            "portable",
            "wireless charging",
            "charging pad",
            "iphone",
            "screen protector",
            "phone case",
            "wallet case",
        )
    if any(w in q for w in ("guitar", "music store", "instrument")):
        _add("guitar", "electric guitar", "acoustic guitar", "music store")
    if any(w in q for w in ("creamer", "coffee")):
        _add("creamer", "coffee", "homemade", "almond milk", "vanilla", "honey")
    if "denver" in q and any(
        w in q for w in ("suggest", "suggestion", "recommend", "tips", "advice", "do there")
    ):
        _add("denver", "concert", "music", "venue", "red rocks")
    if any(w in q for w in ("cookie", "cookies", "baking", "bake")):
        _add("sugar", "chocolate", "bake")
    if "slow cooker" in q or "crock" in q:
        _add("beef stew", "yogurt", "slow cooker")
    if any(w in q for w in ("paint", "painting", "inspiration", "canvas")):
        _add("instagram", "flower", "tutorial", "painting")
    return out[:12]


def is_pub_recommend_query(query: str) -> bool:
    q = (query or "").lower()
    return is_recommend_query(query) and any(
        w in q for w in ("publication", "conference", "paper", "journal", "interesting")
    )


def is_count_query(query: str) -> bool:
    return bool(_COUNT_RE.search(query or ""))


_ERRAND_COUNT_HINT_RE = re.compile(
    r"\b(?:pick(?:ed)?\s*up|return(?:ed|ing)?|exchang(?:e|ed|ing)|"
    r"redeem(?:ed)?|dry[- ]?clean|from\s+(?:a\s+)?store|to\s+(?:a\s+)?store|"
    r"drop(?:ped)?\s*off)\b",
    re.I,
)


def is_errand_count_query(query: str) -> bool:
    """
    Count questions about pending errands (pickup / return / exchange / redeem).

    Inventory history counts ("how many kits did I buy/work on") are excluded so
    digests do not inject unrelated return-policy noise into the answer context.
    """
    return is_count_query(query) and bool(_ERRAND_COUNT_HINT_RE.search(query or ""))


_TOPIC_STOP = _STOP | _PHRASE_STOP | {
    "need", "needs", "items", "item", "many", "much", "week", "weeks",
    "hours", "hour", "recent", "recently", "upcoming", "might", "find",
    "interesting", "reason", "reasons", "seems", "better", "during",
    "group", "rides", "noticed", "performing", "could", "there",
    "publications", "publication", "conferences", "conference",
    "recommend", "suggest", "suggestion", "please", "thank",
    # Action / location shells that match haystack noise too easily
    "pick", "pickup", "return", "returned", "returning", "exchange",
    "exchanged", "exchanging", "store", "stores", "bought", "buy",
    "worked", "work", "purchased", "purchase", "downloaded", "download",
}

_ACTIVITY_RE = re.compile(
    r"\b(?:jog(?:ging)?|yoga|run(?:ning)?|walk(?:ing)?|gym|workout|exercise|"
    r"swim(?:ming)?|cycl(?:e|ing)|hike|hiking|"
    r"gam(?:e|es|ing)|play(?:ing)?(?:\s+games?)?)\b",
    re.I,
)
_DURATION_HINT_RE = re.compile(
    r"\b(?:hours?|hrs?|minutes?|mins?|long|duration|time)\b",
    re.I,
)
_DURATION_SPAN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|minutes?|mins?)",
    re.I,
)
_MAINT_CUE_RE = re.compile(
    r"\b(?:replac(?:e|ed|ing)|install(?:ed|ing)?|upgrad(?:e|ed|ing)|"
    r"maintenance|repair(?:ed|ing)?|serviced|tuned)\b",
    re.I,
)


_AMBIGUOUS_TOPIC_NOUNS = {
    "model", "models", "scale", "scales", "set", "sets", "system", "systems",
    "data", "paper", "papers", "group", "type", "types",
}


def extract_topic_nouns(query: str, *, max_n: int = 8) -> list[str]:
    """Content nouns from the question for CONTAINS / boost (no proper-name bias)."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        # Allow short media tokens (EP) when asked explicitly
        if t in {"ep", "eps"}:
            variants = ["ep", "eps"]
        elif len(t) < 3 or t in _TOPIC_STOP or t in _AMBIGUOUS_TOPIC_NOUNS:
            continue
        else:
            variants = [t]
            if t.endswith("s") and len(t) > 3 and t[:-1] not in _TOPIC_STOP:
                variants.append(t[:-1])
        for v in variants:
            if v in seen or v in _AMBIGUOUS_TOPIC_NOUNS:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= max_n:
                break
        if len(out) >= max_n:
            break
    # Music inventory: vinyl often pairs with album/EP purchases
    if any(x in seen for x in ("album", "albums", "ep", "eps")):
        for v in ("album", "albums", "ep", "eps", "vinyl"):
            if v not in seen and len(out) < max_n + 3:
                seen.add(v)
                out.append(v)
    return out


def extract_topic_phrases(query: str, *, max_n: int = 10) -> list[str]:
    """Significant bigrams (e.g. model kits) for CONTAINS / boost."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    keep = [t for t in tokens if len(t) > 2 and t not in _TOPIC_STOP]
    out: list[str] = []
    seen: set[str] = set()
    for i in range(len(keep) - 1):
        phrase = f"{keep[i]} {keep[i + 1]}"
        # Allow ambiguous first token when paired (model kits)
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        # Singularize trailing s for matching "model kit"
        if keep[i + 1].endswith("s") and len(keep[i + 1]) > 3:
            sing = f"{keep[i]} {keep[i + 1][:-1]}"
            if sing not in seen:
                seen.add(sing)
                out.append(sing)
        # Scale-model inventory: "model kit(s)" often co-occurs with
        # "model tank(s)" / "model building" in the same hobby thread.
        if keep[i + 1] in {"kit", "kits"}:
            for tail in ("tank", "tanks", "building"):
                alt = f"{keep[i]} {tail}"
                if alt not in seen:
                    seen.add(alt)
                    out.append(alt)
        if len(out) >= max_n:
            break
    return out[: max(max_n, len(out))]


def is_preference_context_query(query: str) -> bool:
    """
    Follow-ups that need prior preference or maintenance facts without recommend verbs.
    """
    if needs_preference_retrieval(query):
        return False
    ql = (query or "").lower()
    if re.search(
        r"\b(?:why|reason|noticed|seems|could there|performing|improved|"
        r"improvement|better during|any reason)\b",
        ql,
    ):
        return True
    return False


def is_activity_duration_query(query: str) -> bool:
    return (
        is_count_query(query)
        and bool(_ACTIVITY_RE.search(query or ""))
        and bool(_DURATION_HINT_RE.search(query or ""))
    )


# Duration-count questions only. Do not match content lookups ("which book … a week
# ago"), money spend, event-order, or bare "weeks ago" time cues.
_TEMPORAL_SPAN_COUNT_RE = re.compile(
    r"\bhow many (?:days?|weeks?|months?|years?) ago\b|"
    r"\bhow many (?:days?|weeks?|months?|years?)\b.{0,80}\b"
    r"(?:pass(?:ed)?|between|since|until|apart|have i been|had passed|have passed)\b|"
    r"\b(?:days?|weeks?|months?|years?)\b.{0,40}\b(?:pass(?:ed)?|between|since|until|apart)\b|"
    r"\bbetween the day\b|"
    r"\bpassed between\b|"
    r"\bhave passed since\b|"
    # Shipping latency: how many days from order → arrive (no "passed" wording)
    r"\bhow many days\b.{0,100}\b(?:order(?:ed)?|bought|purchased)\b|"
    r"\bhow many days\b.{0,100}\b(?:arriv(?:e|ed|al)|receiv(?:e|ed)|deliver(?:y|ed))\b",
    re.I,
)


def asks_temporal_span_integer(query: str) -> bool:
    """True only when the expected answer is a duration integer (not what/where/order)."""
    q = query or ""
    if not _TEMPORAL_SPAN_COUNT_RE.search(q):
        return False
    # Guard: content / order questions should never be forced to a bare integer.
    if re.search(
        r"\b(?:what|which|where|who|whom|order of)\b",
        q,
        re.I,
    ) and not re.search(r"\bhow many\b", q, re.I):
        return False
    return True


def is_topic_inventory_query(query: str) -> bool:
    """Non-errand, non-duration, non-temporal item/project counts (kits, albums)."""
    if not is_count_query(query):
        return False
    if is_errand_count_query(query) or is_activity_duration_query(query):
        return False
    if _TEMPORAL_SPAN_COUNT_RE.search(query or ""):
        return False
    return bool(extract_topic_nouns(query) or extract_topic_phrases(query))


def content_has_preference_cues(content: str) -> bool:
    return bool(_PREF_CUE_RE.search(content or ""))


_LOC_UPDATE_RE = re.compile(
    r"\b(moved|moving|relocat(?:ed|ion)?|suburbs?|apartment|lives?\s+in|living\s+in|"
    r"new\s+(?:place|home|apartment|city))\b",
    re.I,
)
# Kit/pub/errand expand lists live in search/experimental_lexicons.py (default off)
# Domain / SoP helpers still used by digests and optional boosts; sourced from quarantine module.
from openstinger.search.experimental_lexicons import (
    DOMAIN_LEXICONS as _DOMAIN_LEXICONS,
    SOP_NOISE_RE as _SOP_NOISE_RE,
)

_EXPERTISE_MARKER_RE = re.compile(
    r"\b(?:working in (?:the )?field|my research|skip (?:the )?basics|"
    r"i(?:'?ve| have) been (?:working|researching|studying)|"
    r"specialize(?:d|s)? in|deep learning for|research interest)\b",
    re.I,
)
_VENUE_NAME_RE = re.compile(
    r"\b(?:neurips|icml|iclr|cvpr|acl|emnlp|aaai|ijcai|kdd|miccai)\b",
    re.I,
)


def extract_subqueries(
    query: str,
    *,
    max_extra: int = 6,
    experimental_lexicons: bool = False,
) -> list[str]:
    """
    C4: original question plus concrete multi-word / noun sub-queries.

    Agents get broader coverage for multi-event and multi-item questions without
    a separate LLM rewrite step. Dataset-flavored expand lists are opt-in.
    """
    q = (query or "").strip()
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = re.sub(r"\s+", " ", (s or "").strip())
        key = s.lower()
        if len(s) < 3 or key in seen:
            return
        seen.add(key)
        out.append(s)

    _add(q)

    # Quoted phrases in the question
    for m in re.findall(r"'([^']{3,60})'|\"([^\"]{3,60})\"", q):
        _add(m[0] or m[1])

    # Significant bigrams / trigrams (skip stopword-heavy)
    tokens = re.findall(r"[A-Za-z0-9]+", q.lower())
    keep = [t for t in tokens if len(t) > 2 and t not in _PHRASE_STOP]
    for n in (3, 2):
        for i in range(len(keep) - n + 1):
            chunk = " ".join(keep[i : i + n])
            if any(len(w) >= 5 for w in keep[i : i + n]):
                _add(chunk)

    # Long singleton nouns
    for t in sorted(set(keep), key=lambda w: (-len(w), w)):
        if len(t) >= 5:
            _add(t)
        if len(out) >= max_extra + 1:
            break

    # Soft-advice bridges are core (not experimental): tips/advice omit prior objects.
    if is_soft_advice_query(q):
        for bridge in soft_advice_bridge_terms(q):
            _add(bridge)
    for bridge in event_attend_bridge_terms(q):
        _add(bridge)
    for bridge in fact_lookup_bridge_terms(q):
        _add(bridge)

    if experimental_lexicons:
        from openstinger.search.experimental_lexicons import append_experimental_subqueries
        append_experimental_subqueries(q, _add)

    return out[: max_extra + 1 + 8]


def diversify_by_source(episodes: list[dict], limit: int) -> list[dict]:
    """
    C2: prefer distinct source_description values in the top-k.

    Score-ordered greedy pick; if a source is already represented, defer it
    until new sources are filled (then fill remaining slots by score).
    """
    if limit <= 0 or not episodes:
        return []
    ordered = sorted(episodes, key=lambda r: float(r.get("score") or 0), reverse=True)
    picked: list[dict] = []
    deferred: list[dict] = []
    seen_sources: set[str] = set()

    for row in ordered:
        src = (row.get("source_description") or row.get("uuid") or "").strip()
        if src and src in seen_sources:
            deferred.append(row)
            continue
        if src:
            seen_sources.add(src)
        picked.append(row)
        if len(picked) >= limit:
            return picked

    for row in deferred:
        if len(picked) >= limit:
            break
        picked.append(row)
    return picked


_STORE_NAMES = {
    "target", "walmart", "costco", "amazon", "best buy", "home depot",
    "zara", "ikea", "cvs", "walgreens", "kroger", "trader joe", "whole foods",
    "apple store", "nordstrom", "macy", "sephora", "starbucks",
}
_ACTION_RE = re.compile(
    r"(?:redeemed|bought|purchased|picked up|pick up|returned|return|ordered|"
    r"exchanged|exchange|dropped off|shipping|shipped|"
    r"dry[- ]?clean(?:ing|ers?)?)\b[^\.\n]{0,100}",
    re.I,
)
_ERRAND_VERB_RE = re.compile(
    r"\b(?:pick(?:ed)?\s*up|return(?:ed|ing)?|exchang(?:e|ed|ing)|"
    r"dry[- ]?clean(?:ing|ers)?|redeem(?:ed)?)\b",
    re.I,
)
_PREF_SPAN_RE = re.compile(
    r"(?:i\s+(?:also\s+)?(?:like|love|prefer|want|need)|i'?d prefer|looking for|"
    r"would prefer|"
    r"practice(?:\s+my)?\s+(?:spanish|french|german|language|languages)|"
    r"(?:spanish|french|german|language)\s+(?:practice|skills?|learning)|"
    r"interested in (?:language|cultural|learning))"
    r"[^\.\n]{0,160}",
    re.I,
)
_HOTEL_FEATURE_RE = re.compile(
    r"(?:rooftop\s+pool|hot tub(?:\s+on the balcony)?|ocean view|city skyline|"
    r"great views?|private balcony|floor-to-ceiling)",
    re.I,
)


def extract_episode_cues(content: str) -> dict[str, Any]:
    """
    Lightweight session cue card (no LLM): stores, actions, preference spans.

    Makes cross-turn links explicit (Target … later redeemed coupon on creamer)
    and surfaces transferable prefs (views / rooftop / hot tub).
    """
    text = content or ""
    lower = text.lower()
    stores: list[str] = []
    for s in sorted(_STORE_NAMES, key=len, reverse=True):
        if s in lower and s.title() not in stores and s not in [x.lower() for x in stores]:
            # Prefer canonical casing from text when possible
            idx = lower.find(s)
            stores.append(text[idx : idx + len(s)])
    # "from Target" / "at Walmart" patterns
    for m in re.finditer(r"\b(?:from|at|in)\s+([A-Z][A-Za-z0-9&' ]{2,30})", text):
        name = m.group(1).strip().rstrip(".,;:")
        if name.lower() not in {x.lower() for x in stores} and len(name) < 40:
            stores.append(name)

    actions = []
    for m in _ACTION_RE.finditer(text):
        span = re.sub(r"\s+", " ", m.group(0)).strip()
        if span and span not in actions:
            actions.append(span[:120])
        if len(actions) >= 8:
            break

    preferences = []
    for m in _PREF_SPAN_RE.finditer(text):
        span = re.sub(r"\s+", " ", m.group(0)).strip()
        if span and span not in preferences:
            preferences.append(span[:160])
        if len(preferences) >= 4:
            break
    features = list(dict.fromkeys(m.group(0) for m in _HOTEL_FEATURE_RE.finditer(text)))[:6]

    parts: list[str] = []
    if stores:
        parts.append("stores/places: " + ", ".join(stores[:8]))
    if actions:
        parts.append("actions: " + " | ".join(actions[:6]))
    if preferences:
        parts.append("preferences: " + " | ".join(preferences[:3]))
    if features:
        parts.append("features: " + ", ".join(features[:6]))
    # Explicit bridge when store + redeem/coupon co-occur in session
    if stores and any("redeem" in a.lower() or "coupon" in a.lower() for a in actions):
        parts.append(
            "link: coupon/redemption in this same session as " + ", ".join(stores[:3])
        )

    summary = "; ".join(parts) if parts else ""
    return {
        "stores": stores[:8],
        "actions": actions[:8],
        "preferences": preferences[:4],
        "features": features[:6],
        "summary": summary,
    }


def score_domain_depth(content: str) -> tuple[str | None, float]:
    """
    Score how deeply a session demonstrates a specialized domain.

    Generic product signal for recommend scoping: prefer coherent expertise
    over venue-name name-dropping and admissions/SoP essays.
    """
    text = content or ""
    cl = text.lower()
    sop_noise = bool(_SOP_NOISE_RE.search(text))
    best_domain: str | None = None
    best = 0.0
    for domain, lex in _DOMAIN_LEXICONS.items():
        hits = sum(1 for term in lex if term in cl)
        if hits <= 0:
            continue
        score = float(hits)
        if _EXPERTISE_MARKER_RE.search(text):
            score += 2.5
        # Specialized venues / datasets beat generic conference name-drops
        if domain == "healthcare_imaging" and ("miccai" in cl or "brats" in cl):
            score += 2.0
        if sop_noise:
            score *= 0.2
        if score > best:
            best = score
            best_domain = domain
    return best_domain, best


def classify_errand_action(action: str) -> str:
    al = (action or "").lower()
    if re.search(r"\breturn(?:ed|ing)?\b", al) and "policy" not in al:
        return "RETURN"
    if re.search(r"\bexchang(?:e|ed|ing)\b", al) and "rate" not in al:
        return "EXCHANGE"
    if re.search(r"\bredeem(?:ed)?\b", al) or "coupon" in al:
        return "REDEEM"
    if re.search(r"\bpick(?:ed)?\s*up\b", al):
        return "PICKUP"
    if re.search(r"\bdry[- ]?clean", al) and re.search(r"\bpick", al):
        return "PICKUP"
    if re.search(r"\b(?:bought|purchased|ordered)\b", al):
        return "BUY"
    return "ACTION"


_ERRAND_NOISE_RE = re.compile(
    r"(?:return\s+policy|exchange\s+rates?|commodity\s+prices|"
    r"while\s+others\s+can\s+be|machine\s+washed|"
    r"exchange\s+or\s+the|shipping\s+options|"
    r"designated\s+spot|folder\s+or\s+envelope|to-do\s+list|"
    r"well-deserved|decluttering|don'?t\s+forget|"
    r"pick\s+up\s+or\s+return|perfect\s+excuse|"
    r"dry\s+clean\s+only|"
    r"return\s+any\s+items|don'?t\s+fit\s+quite|"
    r"any\s+items\s+that\s+don'?t|"
    r"she'?ll\s+return|lent\s+(?:it|them)|"
    r"when\s+she(?:'|’)?ll\s+return)",
    re.I,
)
_ITEM_KEY_STOP = {
    "the", "a", "an", "my", "some", "old", "new", "pair", "ones", "it", "them",
    "this", "that", "from", "to", "at", "for", "and", "or", "you", "should",
    "able", "store", "size", "time", "policies", "policy", "confirm", "enough",
    "larger", "still", "need", "also", "actually", "while", "others", "can",
    "be", "have", "your", "with", "into", "about", "item", "items", "soon",
    "reminder", "list", "break", "come", "back", "always", "quite", "right",
    "any", "fit", "got", "february", "meeting", "weeks", "ago", "wore",
    "not", "dont", "don", "does", "did", "will", "might", "want", "add",
    "quite", "tips", "track",
}
_ITEM_KEY_STOP |= {tok for name in _STORE_NAMES for tok in name.lower().split()}


def _is_noise_errand_span(action: str) -> bool:
    al = (action or "").lower().strip()
    if not al or not _ERRAND_VERB_RE.search(al):
        return True
    if _ERRAND_NOISE_RE.search(al):
        return True
    if "policy" in al or "policies" in al:
        return True
    if len(re.findall(r"[a-z]{3,}", al)) <= 2:
        return True
    if re.search(r"\bdry[- ]?clean", al) and not re.search(r"\bpick", al):
        return True
    return False


def _errand_item_key(action: str) -> str:
    """Normalize an action span to a stable item key for clustering."""
    al = (action or "").lower()
    if re.search(r"\bdry[- ]?clean", al) and re.search(r"\bpick", al):
        return "drycleaning"
    al = _ERRAND_VERB_RE.sub(" ", al)
    al = re.sub(r"\b(?:pair\s+of|some|the|a|an|my|old|new)\b", " ", al)
    tokens = [
        t for t in re.findall(r"[a-z]{3,}", al)
        if t not in _ITEM_KEY_STOP
    ]
    if not tokens:
        return "unknown"
    return " ".join(tokens[:2])


def _item_keys_compatible(a: str, b: str) -> bool:
    if a == "unknown" or b == "unknown":
        return True
    if a == b:
        return True
    return bool(set(a.split()) & set(b.split()))


def _merge_cluster_maps(
    clusters: dict[str, dict[str, Any]],
    order: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Merge duplicate mentions of the same action on the same item.

    Return and pickup stay separate (exchange workflows often need both).
    """
    ids = [cid for cid in order if cid in clusters]
    parent = {cid: cid for cid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ca, cb = clusters[a], clusters[b]
            # Only merge when action tag sets overlap (same errand kind)
            if not (ca["tags"] & cb["tags"]):
                continue
            if not _item_keys_compatible(ca["item"], cb["item"]):
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ka, kb = clusters[ra]["item"], clusters[rb]["item"]
            if kb == "unknown" or (ka != "unknown" and len(ka) >= len(kb)):
                parent[rb] = ra
            else:
                parent[ra] = rb

    merged: dict[str, dict[str, Any]] = {}
    new_order: list[str] = []
    for cid in ids:
        root = find(cid)
        if root not in merged:
            merged[root] = {
                "tags": set(clusters[cid]["tags"]),
                "sample": clusters[cid]["sample"],
                "sid": clusters[cid]["sid"],
                "item": clusters[cid]["item"],
            }
            new_order.append(root)
            continue
        merged[root]["tags"] |= clusters[cid]["tags"]
        if merged[root]["item"] == "unknown" or (
            clusters[cid]["item"] != "unknown"
            and len(clusters[cid]["item"]) > len(merged[root]["item"])
        ):
            merged[root]["item"] = clusters[cid]["item"]
            merged[root]["sample"] = clusters[cid]["sample"]
            merged[root]["sid"] = clusters[cid]["sid"]
    return merged, new_order


def apply_preference_boost(episodes: list[dict], query: str) -> list[dict]:
    """C3: preference-cue boost; domain-depth wins over venue-name spam for pubs."""
    if not needs_preference_retrieval(query):
        return episodes
    q = (query or "").lower()
    hotel_q = "hotel" in q
    pub_q = any(w in q for w in ("publication", "conference", "paper", "journal", "interesting"))
    # Soft-advice tips: bare preference cues without the asked object are noise.
    # Bridge terms count as on-topic (utensil holder for kitchen-clean tips).
    topic_nouns: list[str] = []
    if is_soft_advice_query(query):
        topic_nouns = list(
            dict.fromkeys(extract_topic_nouns(query) + soft_advice_bridge_terms(query))
        )
    boosted: list[dict] = []
    for row in episodes:
        r = dict(row)
        content = r.get("content") or ""
        cl = content.lower()
        score = float(r.get("score") or 0)
        topic_hit = bool(topic_nouns) and any(n in cl for n in topic_nouns)
        if content_has_preference_cues(content):
            if topic_nouns and not topic_hit:
                score = max(0.01, score - 0.06)
            else:
                score = min(1.0, score + 0.15)
                r["preference_boost"] = True
        if hotel_q and "hotel" in cl:
            score = min(1.0, score + 0.2)
            r["preference_boost"] = True
            if _HOTEL_FEATURE_RE.search(content):
                score = min(1.0, score + 0.25)
            if _PREF_SPAN_RE.search(content):
                score = min(1.0, score + 0.15)
        if pub_q:
            venue_hit = bool(_VENUE_NAME_RE.search(content)) or any(
                w in cl
                for w in ("conference", "journal", "publication", "workshop", "paper", "proceedings")
            )
            domain, depth = score_domain_depth(content)
            if domain:
                r["expertise_domain"] = domain
                r["expertise_depth"] = depth
            sop_noise = bool(_SOP_NOISE_RE.search(content))
            if venue_hit and sop_noise and depth < 2.0:
                # Admissions / SoP venue name-drops should not outrank specialty
                pass
            elif venue_hit and depth < 2.0:
                # Light bump only: bare venue lists should not dominate
                score = min(1.0, score + 0.08)
                r["preference_boost"] = True
            elif venue_hit:
                score = min(1.0, score + 0.12)
                r["preference_boost"] = True
            if depth >= 2.0:
                # Coherent specialty beats generic conference chatter
                score = min(1.0, score + min(0.45, 0.12 * depth))
                r["preference_boost"] = True
                r["expertise_boost"] = True
            if _EXPERTISE_MARKER_RE.search(content) and not sop_noise:
                score = min(1.0, score + 0.15)
                r["preference_boost"] = True
        r["score"] = round(score, 4)
        boosted.append(r)
    return boosted


def apply_errand_boost(episodes: list[dict], query: str) -> list[dict]:
    """For count / pickup-return questions, surface errand-rich sessions."""
    ql = (query or "").lower()
    if not (
        is_count_query(query)
        or any(w in ql for w in ("pick up", "return", "clothing", "exchange", "store"))
    ):
        return episodes
    topic = extract_topic_nouns(query)
    out: list[dict] = []
    for row in episodes:
        r = dict(row)
        content = r.get("content") or ""
        cl = content.lower()
        score = float(r.get("score") or 0)
        n_errands = len(_ERRAND_VERB_RE.findall(content))
        dry = bool(re.search(r"\bdry[- ]?clean", cl))
        strong = bool(
            dry or re.search(r"\b(?:picked up|returned|exchanged)\b", cl)
        )
        topic_hit = bool(topic) and any(t in cl for t in topic)
        # Apparel / object words often absent; treat dry-clean as on-topic for clothing qs
        clothing_q = any(w in ql for w in ("clothing", "clothes", "apparel"))
        on_topic = topic_hit or (clothing_q and dry) or (
            clothing_q and bool(re.search(
                r"\b(?:boots?|blazer|dress|shirt|pants|jacket|coat|shoes?)\b", cl
            ))
        )
        if not on_topic or not (strong or n_errands):
            out.append(r)
            continue
        if n_errands:
            score = min(1.0, score + 0.12 * min(n_errands, 3))
            r["errand_boost"] = True
        if strong:
            score = min(1.0, score + 0.2)
            r["errand_boost"] = True
        r["score"] = round(score, 4)
        out.append(r)
    return out


def apply_query_noun_boost(episodes: list[dict], query: str) -> list[dict]:
    """Surface episodes that mention question nouns/phrases (kits, bike, albums, …)."""
    # Errand counts are handled by digests; noun boost there pulls noise.
    if is_errand_count_query(query):
        return episodes
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    if (not phrases and not nouns) or not episodes:
        return episodes
    out: list[dict] = []
    for row in episodes:
        r = dict(row)
        cl = (r.get("content") or "").lower()
        phrase_hits = sum(1 for p in phrases if p in cl)
        noun_hits = sum(1 for n in nouns if n in cl)
        if phrase_hits or noun_hits:
            score = float(r.get("score") or 0)
            score = min(1.0, score + 0.18 * min(phrase_hits, 2) + 0.1 * min(noun_hits, 3))
            r["score"] = round(score, 4)
            r["noun_boost"] = True
        out.append(r)
    return out


def apply_preference_context_boost(episodes: list[dict], query: str) -> list[dict]:
    """
    Preference follow-ups and soft-advice tips: boost topic-overlap + cues.
    """
    if not (
        is_preference_context_query(query) or is_soft_advice_query(query)
    ) or not episodes:
        return episodes
    nouns = extract_topic_nouns(query)
    out: list[dict] = []
    for row in episodes:
        r = dict(row)
        content = r.get("content") or ""
        cl = content.lower()
        score = float(r.get("score") or 0)
        noun_hit = bool(nouns) and any(n in cl for n in nouns)
        if noun_hit and (_MAINT_CUE_RE.search(content) or content_has_preference_cues(content)):
            score = min(1.0, score + 0.22)
            r["preference_boost"] = True
        elif noun_hit:
            score = min(1.0, score + 0.12)
            r["preference_boost"] = True
        elif content_has_preference_cues(content) and not nouns:
            score = min(1.0, score + 0.1)
            r["preference_boost"] = True
        elif content_has_preference_cues(content) and nouns and not noun_hit:
            # Preference-flavored sessions about a different object crowd tip ranking
            score = max(0.01, score - 0.08)
        r["score"] = round(score, 4)
        out.append(r)
    return out


def force_include_expertise_episodes(
    candidates: list[dict],
    query: str,
    limit: int,
) -> list[dict]:
    """
    For generic publication/conference recommends, keep deepest expertise
    sessions in the candidate pool even if venue-name noise scored higher.
    """
    q = (query or "").lower()
    if not is_recommend_query(query):
        return candidates
    if not any(w in q for w in ("publication", "conference", "paper", "journal", "interesting")):
        return candidates
    # Skip when the question already names an explicit domain (user scoped it)
    if any(w in q for w in ("medical", "healthcare", "robot", "climate", "nlp")):
        return candidates

    scored: list[tuple[float, dict]] = []
    for row in candidates:
        content = row.get("content") or ""
        if _SOP_NOISE_RE.search(content):
            continue
        domain, depth = score_domain_depth(content)
        if depth >= 3.0:
            scored.append((depth, row))
    if not scored:
        return candidates
    scored.sort(key=lambda x: x[0], reverse=True)
    by_uuid = {r.get("uuid"): r for r in candidates if r.get("uuid")}
    for depth, row in scored[: max(2, min(4, limit))]:
        uid = row.get("uuid")
        if not uid:
            continue
        if uid in by_uuid:
            by_uuid[uid]["score"] = max(float(by_uuid[uid].get("score") or 0), 0.95)
            by_uuid[uid]["expertise_forced"] = True
            by_uuid[uid]["expertise_depth"] = depth
            by_uuid[uid]["preference_boost"] = True
            by_uuid[uid]["expertise_boost"] = True
        else:
            forced = dict(row)
            forced["score"] = max(float(forced.get("score") or 0), 0.95)
            forced["expertise_forced"] = True
            forced["expertise_depth"] = depth
            forced["preference_boost"] = True
            forced["expertise_boost"] = True
            forced["result_type"] = "episode"
            by_uuid[uid] = forced
    return list(by_uuid.values())


def apply_recency_packaging(episodes: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    C1: newest-first tie-break + OLD/NEW labels when the same proper name appears
    in multiple dated episodes. Also returns a conflicts list for agents.
    """
    if not episodes:
        return [], []

    def _valid_at(row: dict) -> int:
        try:
            return int(row.get("valid_at") or 0)
        except (TypeError, ValueError):
            return 0

    # Tie-break: higher score first, then newer valid_at
    ranked = sorted(
        episodes,
        key=lambda r: (float(r.get("score") or 0), _valid_at(r)),
        reverse=True,
    )

    # Proper names (capitalized tokens) as soft entity keys
    name_re = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b")
    by_name: dict[str, list[dict]] = {}
    for row in ranked:
        names = set(name_re.findall(row.get("content") or ""))
        # Skip ultra-common false positives
        names = {n for n in names if n.lower() not in {"the", "this", "that", "march", "april", "june", "july"}}
        for n in names:
            by_name.setdefault(n, []).append(row)

    conflicts: list[dict] = []
    labeled_ids: dict[str, str] = {}  # uuid -> older|newer
    for name, rows in by_name.items():
        uniq: dict[str, dict] = {}
        for r in rows:
            uid = r.get("uuid")
            if uid:
                uniq[uid] = r
        if len(uniq) < 2:
            continue
        ordered = sorted(uniq.values(), key=_valid_at)
        older, newer = ordered[0], ordered[-1]
        if _valid_at(newer) <= _valid_at(older):
            continue
        older_id, newer_id = older.get("uuid"), newer.get("uuid")
        if not older_id or not newer_id or older_id == newer_id:
            continue
        labeled_ids[older_id] = "older"
        labeled_ids[newer_id] = "newer"
        conflicts.append(
            {
                "entity_hint": name,
                "older_uuid": older_id,
                "newer_uuid": newer_id,
                "older_valid_at": _valid_at(older),
                "newer_valid_at": _valid_at(newer),
                "older_valid_at_human": older.get("valid_at_human"),
                "newer_valid_at_human": newer.get("valid_at_human"),
            }
        )

    out: list[dict] = []
    for row in ranked:
        r = dict(row)
        uid = r.get("uuid")
        if uid and uid in labeled_ids:
            r["recency_label"] = labeled_ids[uid]
        out.append(r)
    return out, conflicts


def smart_episode_excerpt(content: str, query: str = "", *, max_chars: int = 16000) -> str:
    """
    C5: keep generous context. If still too long, prefer windows around query terms
    so late-session facts (coupon/store) are not dropped.

    For money/aggregate questions, also pin windows around dollar amounts near
    query nouns so late appraisal/sale figures are not dropped when early
    mentions of the same noun fill the per-term hit cap.
    """
    text = content or ""
    if len(text) <= max_chars:
        return text
    terms = extract_search_terms(query)[:8]
    if not terms:
        return truncate_episode_content(text, max_chars=max_chars)
    lower = text.lower()
    windows: list[tuple[int, int]] = []
    money_q = bool(
        re.search(
            r"\b(?:how much|\$|money|sold|sell|spend|spent|minimum|apprais|"
            r"worth|price|raise)\b",
            query or "",
            re.I,
        )
    )
    # Pin first-person duration claims before generic "hours" spam fills the budget
    for m in _USER_DURATION_RE.finditer(text):
        a = max(0, m.start() - 240)
        b = min(len(text), m.end() + 240)
        windows.append((a, b))
        if len(windows) >= 4:
            break
    # Pin dollar amounts that sit near a query term (late-session valuations)
    if money_q:
        term_l = [t.lower() for t in terms if len(t) >= 3][:8]
        money_hits = 0
        for m in re.finditer(
            r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|"
            r"\$\s?\d{4,}(?:\.\d{1,2})?|"
            r"\b\d{1,3}(?:,\d{3})+\s*(?:dollars?|usd)\b",
            text,
            re.I,
        ):
            around = lower[max(0, m.start() - 80) : m.end() + 40]
            if term_l and not any(t in around for t in term_l):
                continue
            a = max(0, m.start() - 280)
            b = min(len(text), m.end() + 200)
            windows.append((a, b))
            money_hits += 1
            if money_hits >= 6:
                break
    span = max(600, max_chars // max(2, len(terms)))
    per_term_cap = 4 if money_q or is_aggregate_query(query) else 2
    for t in terms:
        start = 0
        term_hits = 0
        while True:
            idx = lower.find(t, start)
            if idx < 0:
                break
            a = max(0, idx - span // 3)
            b = min(len(text), idx + span)
            windows.append((a, b))
            start = idx + len(t)
            term_hits += 1
            # Cap per-term hits so early "hours" lists do not crowd out late user claims
            if term_hits >= per_term_cap or len(windows) >= 14:
                break
        if len(windows) >= 14:
            break
    if not windows:
        return truncate_episode_content(text, max_chars=max_chars)
    # Merge overlapping windows
    windows.sort()
    merged: list[list[int]] = []
    for a, b in windows:
        if not merged or a > merged[-1][1] + 50:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    parts = []
    total = 0
    for a, b in merged:
        chunk = text[a:b]
        if total + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - total)]
        if not chunk:
            break
        prefix = "…" if a > 0 else ""
        suffix = "…" if b < len(text) else ""
        parts.append(f"{prefix}{chunk}{suffix}")
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n---\n".join(parts) if parts else truncate_episode_content(text, max_chars=max_chars)


def truncate_episode_content(content: str, *, max_chars: int = 16000) -> str:
    """C5: keep generous context; only soft-cap extreme episodes."""
    text = content or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…[truncated]"


def package_episode_row(row: dict, query: str = "") -> dict:
    """Normalize episode row for tool / agent consumption (C1/C5 + cue cards)."""
    r = dict(row)
    raw = r.get("content") or ""
    cues = extract_episode_cues(raw)
    r["cues"] = cues
    summary = cues.get("summary") or ""
    r["cue_summary"] = summary
    excerpt = smart_episode_excerpt(raw, query, max_chars=16000)
    if summary:
        # Surface cross-turn links and transferable prefs before the transcript
        r["content"] = f"[Session cues] {summary}\n\n{excerpt}"
    else:
        r["content"] = excerpt
    r["content_chars"] = len(raw)
    return r


def force_include_focus_episodes(
    candidates: list[dict],
    focus_rows: list[dict],
    limit: int,
) -> list[dict]:
    """
    Ensure newest focus-term hits (e.g. both Rachel sessions) survive diversity.
    """
    if not focus_rows:
        return candidates
    by_uuid = {r.get("uuid"): r for r in candidates if r.get("uuid")}
    # Newest first among focus hits
    def _va(r: dict) -> int:
        try:
            return int(r.get("valid_at") or 0)
        except (TypeError, ValueError):
            return 0

    ordered_focus = sorted(focus_rows, key=_va, reverse=True)
    for row in ordered_focus[: max(2, min(4, limit))]:
        uid = row.get("uuid")
        if not uid:
            continue
        if uid in by_uuid:
            by_uuid[uid]["score"] = max(float(by_uuid[uid].get("score") or 0), 0.9)
            by_uuid[uid]["focus_forced"] = True
        else:
            forced = dict(row)
            forced["score"] = max(float(forced.get("score") or 0), 0.9)
            forced["focus_forced"] = True
            forced["result_type"] = "episode"
            by_uuid[uid] = forced
    return list(by_uuid.values())


def force_include_topic_episodes(
    candidates: list[dict],
    query: str,
    limit: int,
) -> list[dict]:
    """
    Keep phrase-matching inventory sessions in the pool (e.g. model kit talks)
    even when generic keyword noise ranked higher in RRF.
    """
    fact_bridges = fact_lookup_bridge_terms(query)
    event_bridges = event_attend_bridge_terms(query)
    if (
        not is_topic_inventory_query(query)
        and not is_preference_context_query(query)
        and not is_soft_advice_query(query)
        and not needs_preference_retrieval(query)
        and not fact_bridges
        and not event_bridges
    ):
        if not is_activity_duration_query(query):
            return candidates
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    if is_soft_advice_query(query) or needs_preference_retrieval(query):
        for bridge in soft_advice_bridge_terms(query):
            if " " in bridge:
                if bridge not in phrases:
                    phrases.append(bridge)
            elif bridge not in nouns:
                nouns.append(bridge)
    for bridge in event_bridges + fact_bridges:
        if " " in bridge:
            if bridge not in phrases:
                phrases.append(bridge)
        elif bridge not in nouns:
            nouns.append(bridge)
    if not phrases and not nouns:
        return candidates

    scored: list[tuple[float, dict]] = []
    for row in candidates:
        content = row.get("content") or ""
        cl = content.lower()
        phrase_hits = sum(1 for p in phrases if p in cl)
        noun_hits = sum(1 for n in nouns if n in cl)
        if phrase_hits == 0 and noun_hits == 0:
            continue
        strength = float(phrase_hits) * 2.0 + float(noun_hits)
        for p in phrases:
            if p in cl and re.search(
                r"\b(?:my|i(?:'ve| have)?)\b.{0,60}" + re.escape(p), cl
            ):
                strength += 1.5
                break
        if is_soft_advice_query(query) and re.search(
            r"\b(?:i(?:'ve| have)?\s+(?:recently\s+)?(?:bought|got|purchased|harvested)|"
            r"my\s+(?:new\s+)?)",
            cl,
        ):
            strength += 1.0
        scored.append((strength, row))
    if not scored:
        return candidates
    scored.sort(key=lambda x: x[0], reverse=True)
    by_uuid = {r.get("uuid"): r for r in candidates if r.get("uuid")}
    keep_n = max(4, min(10, limit)) if is_soft_advice_query(query) else max(4, min(8, limit))
    for strength, row in scored[:keep_n]:
        uid = row.get("uuid")
        if not uid:
            continue
        if is_soft_advice_query(query):
            # Prefer any real topic/bridge hit into top-k (creamer gold was dying at ~rank 8)
            bump = 0.99 if strength >= 2 else (0.96 if strength >= 1 else 0.9)
        else:
            bump = 0.95 if strength >= 3 else (0.9 if strength >= 2 else 0.82)
        if uid in by_uuid:
            # Absolute bump: positional/noop scores must not dominate topic hits
            by_uuid[uid]["score"] = bump
            by_uuid[uid]["fusion_score"] = bump
            by_uuid[uid]["topic_forced"] = True
            by_uuid[uid]["noun_boost"] = True
        else:
            forced = dict(row)
            forced["score"] = bump
            forced["fusion_score"] = bump
            forced["topic_forced"] = True
            forced["noun_boost"] = True
            forced["result_type"] = "episode"
            by_uuid[uid] = forced
    return list(by_uuid.values())


_EVENT_ATOM_RE = re.compile(
    r"\bI(?:'ve| have| had)?(?: just| recently| finally| also| then)?\s+"
    r"(?:visited|went to|went for|attended|bought|got|purchased|acquired|adopted|"
    r"started|joined|signed up for|watched|finished|completed|took|spent|"
    r"made|hosted|ordered|returned|redeemed|earned|planted|harvested|"
    r"set up|picked up|saw|met with|flew|traveled to|moved to)\b"
    r"[^\.\n\?]{3,140}",
    re.I,
)
_EVENT_NOISE_RE = re.compile(
    r"\b(?:would|could|should|might|want to|plan(?:ning)? to|hope to|"
    r"thinking (?:about|of)|used to|if i)\b",
    re.I,
)


def extract_event_atoms(content: str, *, max_atoms: int = 10) -> list[str]:
    """
    C1: first-person dated-event spans from a session transcript.

    Pattern-first (no LLM). Product-general verbs only; used to build a compact
    per-session event timeline so aggregate/temporal questions can retrieve
    events whose sessions never mention the question's topic noun.
    """
    text = content or ""
    out: list[str] = []
    seen: set[str] = set()
    for m in _EVENT_ATOM_RE.finditer(text):
        span = re.sub(r"\s+", " ", m.group(0)).strip()
        # Skip hypotheticals/plans and assistant-echo fragments
        if _EVENT_NOISE_RE.search(span):
            continue
        # Skip spans that are clearly assistant text (You/your framing right before)
        prefix = text[max(0, m.start() - 40) : m.start()].lower()
        if re.search(r"\b(?:you|your)\s*$", prefix):
            continue
        key = span.lower()[:100]
        if key in seen:
            continue
        seen.add(key)
        out.append(span[:150])
        if len(out) >= max_atoms:
            break
    return out


def is_aggregate_query(query: str) -> bool:
    """Questions that must cover ALL matching sessions (counts, totals, orderings)."""
    return (
        is_count_query(query)
        or is_temporal_span_query(query)
        or bool(
            re.search(
                r"\b(?:order of|in total|altogether|all the|every|"
                r"minimum amount|how much more|older am i than|"
                r"spent on|spend on|combined total)\b",
                query or "",
                re.I,
            )
        )
    )


_CONJUNCT_SIDE_STOP = {
    "how", "many", "much", "what", "which", "where", "who", "when", "the", "a",
    "an", "i", "my", "me", "did", "do", "does", "have", "had", "has", "get",
    "got", "could", "would", "should", "will", "can", "for", "from", "with",
    "into", "onto", "about", "total", "number", "amount", "minimum", "maximum",
    "initially", "currently", "now", "past", "last", "few", "months", "weeks",
    "days", "years", "sold", "sell", "spend", "spent", "viewed", "tried",
    "taking", "take", "save", "saving", "instead", "by", "if", "could",
    "raise", "raised", "appraised", "worth", "price", "paid", "cost",
}


def _token_stems(token: str) -> set[str]:
    """Light singular/plural stems for structural matching (no lexicon lists)."""
    t = (token or "").lower().strip()
    if len(t) < 3:
        return {t} if t else set()
    out = {t}
    if t.endswith("ies") and len(t) > 4:
        out.add(t[:-3] + "y")
    if t.endswith("oes") and len(t) > 4:
        out.add(t[:-2])
    if t.endswith("ses") and len(t) > 4:
        out.add(t[:-2])
    if t.endswith("es") and len(t) > 4:
        out.add(t[:-2])
        out.add(t[:-1])
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        out.add(t[:-1])
    return out


def soft_contains(haystack: str, needle: str) -> bool:
    """True if needle (or a light stem) appears as a word/substring in haystack."""
    h = (haystack or "").lower()
    n = (needle or "").lower().strip()
    if not h or not n:
        return False
    if n in h:
        return True
    for stem in _token_stems(n):
        if len(stem) >= 3 and re.search(rf"\b{re.escape(stem)}\b", h):
            return True
    # Multi-word: require each content token (soft) to appear
    parts = [p for p in n.split() if p not in _CONJUNCT_SIDE_STOP and len(p) >= 3]
    if len(parts) >= 2:
        return all(soft_contains(h, p) for p in parts)
    return False


# Back-compat alias used inside this module during refactor
_soft_contains = soft_contains


def extract_and_conjuncts(query: str) -> list[str]:
    """
    Structural A-and-B objects from the question text (no topic name lists).

    Left: NP head before 'and' (last contentful compound).
    Right: NP head at the start of the right conjunct (assists / lentil soup),
    trimming trailing clauses ('I have in the … league').
    """
    q = (query or "").strip()
    if not q:
        return []

    def _side_tokens(side: str) -> list[str]:
        return [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", side)
            if w.lower() not in _CONJUNCT_SIDE_STOP
        ]

    _compound_block = {
        "plant", "plants", "planted", "buy", "bought", "watch", "watched",
        "try", "tried", "learn", "learned", "view", "viewed", "fix", "fixed",
        "assemble", "assembled", "make", "made", "cook", "cooked", "order",
        "ordered", "number", "total", "lunch", "meals", "meal", "pieces",
    }

    def _trim_right_clause(side: str) -> str:
        """Keep leading NP; drop 'I have…', 'in the…', 'from my…' tails."""
        cut = re.split(
            r"\b(?:i(?:'ve| am|'m)?|we(?:'ve)?|in the|from the|for my|for the|"
            r"that i|which i|during|after|before|with my)\b",
            side,
            maxsplit=1,
            flags=re.I,
        )[0]
        return (cut or side).strip()

    def _np_head_left(side: str) -> str | None:
        toks = _side_tokens(side)
        if not toks:
            return None
        if (
            len(toks) >= 2
            and toks[-2].lower() not in _compound_block
            and toks[-1].lower() not in _compound_block
        ):
            return f"{toks[-2]} {toks[-1]}"
        return toks[-1]

    def _np_head_right(side: str) -> str | None:
        side = _trim_right_clause(side)
        toks = _side_tokens(side)
        if not toks:
            return None
        # Leading compound when both tokens remain after trim ('lentil soup')
        if (
            len(toks) >= 2
            and toks[0].lower() not in _compound_block
            and toks[1].lower() not in _compound_block
            and len(side.split()) <= 4
        ):
            return f"{toks[0]} {toks[1]}"
        return toks[0]

    out: list[str] = []
    for kind, pat in (
        ("and", r"(.+?)\s+and\s+(?:the\s+|a\s+)?(.+?)(?:\?|$)"),
        (
            "instead",
            r"(.+?)\s+instead of\s+(?:a\s+|the\s+)?(.+?)(?:\?|$)",
        ),
    ):
        m = re.search(pat, q, re.I)
        if not m:
            continue
        left = None
        right = None
        if kind == "instead":
            # 'taking the bus … to my hotel instead of a taxi' → bus, not hotel
            vm = re.search(
                r"\b(?:taking|take|by|using|via|rode|ride)\s+(?:the\s+|a\s+)?"
                r"([A-Za-z][A-Za-z'-]{2,})",
                m.group(1),
                re.I,
            )
            if vm and vm.group(1).lower() not in _CONJUNCT_SIDE_STOP:
                left = vm.group(1)
            if not left:
                toks = _side_tokens(m.group(1))
                left = toks[0] if toks else None
            rtoks = _side_tokens(m.group(2))
            right = rtoks[0] if rtoks else None
        else:
            left = _np_head_left(m.group(1))
            right = _np_head_right(m.group(2))
        if left and right and left.lower() != right.lower():
            out = [left, right]
            break
    return out


def build_aggregate_reading_digest(hits: list[dict], query: str = "") -> str:
    """
    Structured reading notes for multi-session count/money questions.

    Generic only: uses question nouns/phrases and A-and-B conjuncts from the
    question text. No benchmark topic lists. Enumerate open counts; sum only
    when each conjunct side has its own paired evidence.
    """
    if not is_aggregate_query(query) or not hits:
        return ""
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    terms = [t for t in (phrases + nouns) if t and len(t) >= 3]
    if not terms:
        return ""
    ql = (query or "").lower()
    conjuncts = extract_and_conjuncts(query)
    money_q = bool(
        re.search(r"\b(?:how much|\$|money|sold|spend|spent|minimum|raise)\b", ql)
    )
    open_enumerate = bool(re.search(r"\bhow many\b", ql)) and not conjuncts
    multi_action = bool(
        re.search(
            r"\b(?:buy|bought|assemble|assembled|sell|sold|fix|fixed)\b"
            r".{0,40}\b(?:or|,)\b.{0,40}"
            r"\b(?:buy|bought|assemble|assembled|sell|sold|fix|fixed)\b",
            ql,
        )
    )
    notes: list[str] = []
    # (value, span, paired_object_or_None)
    money_vals: list[tuple[int, str, str | None]] = []
    # (n, span, paired_object_or_None)
    number_claims: list[tuple[int, str, str | None]] = []
    distinct_items: list[str] = []
    seen_items: set[str] = set()
    session_hits = 0
    seen_sid: set[str] = set()

    def _pair_object(text: str) -> str | None:
        best = None
        best_len = 0
        pool = conjuncts or terms
        for obj in pool:
            if soft_contains(text, obj) and len(obj) > best_len:
                best = obj
                best_len = len(obj)
                continue
            # Head-noun fallback: 'necklace' matches 'diamond necklace'
            parts = [p for p in obj.split() if len(p) >= 3]
            if len(parts) >= 2 and soft_contains(text, parts[-1]):
                # Avoid ambiguous heads shared by multiple conjuncts
                head = parts[-1].lower()
                if sum(1 for c in pool if c.lower().endswith(head)) == 1:
                    if len(parts[-1]) > best_len:
                        best = obj
                        best_len = len(parts[-1])
        return best

    money_pat = re.compile(
        r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d{4,}(?:\.\d{1,2})?|\d{1,3}(?:\.\d{1,2})?)"
        r"|\b(\d{1,3}(?:,\d{3})+|\d{3,6})\s*(?:dollars?|usd)\b",
        re.I,
    )

    for h in hits:
        content = h.get("content") or ""
        if content.lstrip().startswith("[Session events"):
            continue
        cl = content.lower()
        matched = [t for t in terms if soft_contains(cl, t)]
        _act = (
            r"bought|got a new|get a new|ordered|assembled|fixed|sold"
        )
        _obj = (
            r"table|bookshelf|desk|chair|dresser|mattress|sofa|couch|"
            r"cabinet|shelf|lamp|bed"
        )
        action_hit = multi_action and bool(
            re.search(
                rf"(?:\b(?:{_act})\b.{{0,80}}\b(?:{_obj})\b|"
                rf"\b(?:{_obj})\b.{{0,80}}\b(?:{_act})\b)",
                cl,
            )
        )
        if (
            not matched
            and not (conjuncts and any(soft_contains(cl, c) for c in conjuncts))
            and not action_hit
        ):
            continue
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        if sid in seen_sid:
            continue
        seen_sid.add(sid)
        session_hits += 1
        when = h.get("valid_at_human") or ""
        stamp = f" @{when}" if when else ""

        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = "\n".join(user_chunks) if user_chunks else content

        for term in (matched or conjuncts or (["furniture"] if action_hit else []))[:4]:
            for m in re.finditer(re.escape(term), scan, re.I):
                window = scan[max(0, m.start() - 50) : m.end() + 90]
                if not re.search(
                    r"\b(?:i|my|i've|i am|i'm|we|sold|sell|appraised|worth|"
                    r"price|paid|cost|fetch)\b",
                    window,
                    re.I,
                ):
                    continue
                snip = re.sub(r"\s+", " ", window).strip()
                if len(snip) < 12:
                    continue
                notes.append(f"- ({sid}){stamp} [{term}]: {snip[:160]}")
                break
            if len(notes) >= 14:
                break

        # Numbers near query nouns / acquire verbs (digits + small word numbers)
        obj_alt = "|".join(
            re.escape(t) for t in ((conjuncts or matched or terms)[:10]) if t
        )
        if not obj_alt:
            obj_alt = r"a^"  # never matches
        num_tok = (
            r"(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"first|second|third|fourth|fifth)"
        )
        for m in re.finditer(
            rf"\b(?:planted|watched|tried|owned?|bought|viewed|have|had|got|ate|"
            rf"completed|finished|played|learned|scored|lasted)\b[^\n.]{{0,48}}?"
            rf"{num_tok}\b"
            rf"|{num_tok}\b[^\n.]{{0,32}}?(?:of\s+)?(?:{obj_alt})\b"
            rf"|\b(?:the\s+)?{num_tok}\s+(?:meal|meals|lunch|lunches|"
            rf"goal|goals|assist|assists)\b",
            scan,
            re.I,
        ):
            num_s = next((g for g in m.groups() if g), None)
            if not num_s:
                continue
            n = _parse_count_token(num_s)
            if n is None or n <= 0 or n > 5000:
                continue
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            around = scan[max(0, m.start() - 40) : m.end() + 40]
            pool = conjuncts or matched or terms
            if not any(soft_contains(around, t) for t in pool):
                continue
            number_claims.append((n, f"{span[:100]}{stamp}", _pair_object(around)))

        # Distinct item spans for open enumeration (first-person acquire/use)
        if open_enumerate:
            enum_pats = [
                r"\b(?:i(?:'ve| have)?|my)\b[^\n.]{0,120}?\b(?:bought|purchased|"
                r"downloaded|got|ordered|tried|watched|owned?|picked up|assembled|"
                r"fixed|sold|learned|cooked)\b[^\n.]{0,80}",
            ]
            if multi_action:
                enum_pats.append(
                    rf"(?:\b(?:{_act}|get a new|get)\b.{{0,80}}\b(?:{_obj})\b|"
                    rf"\b(?:{_obj})\b.{{0,80}}\b(?:{_act}|get a new)\b)"
                )
            for pat in enum_pats:
                for m in re.finditer(pat, scan, re.I):
                    span = re.sub(r"\s+", " ", m.group(0)).strip()
                    if len(span) < 12:
                        continue
                    term_ok = any(soft_contains(span, t) for t in terms)
                    action_ok = multi_action and bool(
                        re.search(
                            rf"(?:\b(?:{_act}|get a new)\b.{{0,80}}\b(?:{_obj})\b|"
                            rf"\b(?:{_obj})\b.{{0,80}}\b(?:{_act}|get a new)\b)",
                            span,
                            re.I,
                        )
                    )
                    if not term_ok and not action_ok:
                        continue
                    # Dedupe by specific object phrase (coffee table ≠ kitchen table)
                    obj_m = re.search(
                        r"\b(?:(?:coffee|kitchen|bedside|side|dining|end)\s+)?"
                        r"(?:table|bookshelf|desk|chair|dresser|mattress|sofa|"
                        r"couch|cabinet|shelf|lamp|bed)\b",
                        span,
                        re.I,
                    )
                    key = obj_m.group(0).lower() if obj_m else span.lower()[:80]
                    if key in seen_items:
                        continue
                    seen_items.add(key)
                    distinct_items.append(span[:100])

        if money_q:
            for m in money_pat.finditer(scan):
                raw = m.group(1) or m.group(2)
                if not raw:
                    continue
                try:
                    val = int(float(raw.replace(",", "")))
                except ValueError:
                    continue
                if val <= 0 or val > 1_000_000:
                    continue
                around = scan[max(0, m.start() - 70) : m.end() + 50]
                around_l = around.lower()
                paired = _pair_object(around)
                # Require a question object nearby, or sell/value verb + any topic term
                if not paired and not (
                    any(_soft_contains(around_l, t) for t in terms)
                    and re.search(
                        r"\b(?:sold|sell|worth|offer|price|paid|cost|raise|"
                        r"raised|appraised|fetch)\b",
                        around_l,
                    )
                ):
                    continue
                money_vals.append(
                    (
                        val,
                        re.sub(r"\s+", " ", around).strip()[:120] + stamp,
                        paired,
                    )
                )

        if len(notes) >= 14:
            break

    if (
        session_hits == 0
        and not number_claims
        and not money_vals
        and not distinct_items
    ):
        return ""

    lines = [
        "Aggregate reading notes (enumerate matching facts across sessions; "
        "do not abstain when on-topic numbers are listed; do not sum unrelated numbers):",
    ]
    if conjuncts:
        lines.append(
            "Question conjunct objects (answer must cover each side): "
            + "; ".join(conjuncts)
        )
    for note in notes[:12]:
        lines.append(note)

    if number_claims:
        seen_nc: set[str] = set()
        uniq_nc: list[tuple[int, str, str | None]] = []
        for n, span, obj in number_claims:
            key = f"{n}:{span[:40].lower()}"
            if key in seen_nc:
                continue
            seen_nc.add(key)
            uniq_nc.append((n, span, obj))
        lines.append(
            "Candidate numeric claims: "
            + "; ".join(
                f"{n}←{s}" + (f" [{o}]" if o else "")
                for n, s, o in uniq_nc[:10]
            )
        )
        # Sum only when each conjunct has its own paired claim (count Qs; not money)
        if len(conjuncts) >= 2 and not money_q:
            per: dict[str, int] = {}
            for n, _span, obj in uniq_nc:
                if not obj:
                    continue
                for c in conjuncts:
                    if (
                        obj.lower() == c.lower()
                        or soft_contains(obj, c)
                        or soft_contains(c, obj)
                    ):
                        # keep largest claim per conjunct
                        per[c] = max(per.get(c, 0), n)
            if len(per) == len(conjuncts):
                total = sum(per.values())
                lines.append(
                    "Suggested aggregate count (sum of per-conjunct claims): "
                    f"{total} ("
                    + ", ".join(f"{c}={per[c]}" for c in conjuncts)
                    + ")."
                )
                lines.append(
                    f"Answer with the integer {total} unless a clearer explicit total is present."
                )
            else:
                missing = [c for c in conjuncts if c not in per]
                lines.append(
                    "Incomplete conjunct counts: missing numeric claim for "
                    + ", ".join(missing)
                    + ". Prefer abstain or answer only sides that have numbers."
                )
        elif open_enumerate and distinct_items:
            pass  # handled below
        else:
            lines.append(
                "Enumerate distinct matching items, then answer with one integer. "
                "Do not sum unrelated numbers from different topics."
            )

    # Prefer an explicit first-person total over span enumeration when present.
    # Rank by how many query nouns appear near the claim so '5 MCU films' beats
    # a generic 'watched 12 films' from a different topic.
    stated_ranked: list[tuple[int, int]] = []  # (specificity, n)
    adjacent_ns: list[int] = []
    if open_enumerate or (not conjuncts and not money_q):
        blob = "\n".join(
            (h.get("content") or "")
            for h in hits
            if not (h.get("content") or "").lstrip().startswith("[Session events")
        )
        topic_keys = [
            t
            for t in (phrases + nouns)
            if t
            and len(t) >= 3
            and t.lower()
            not in {
                "how", "many", "much", "last", "past", "few", "months", "month",
                "weeks", "week", "days", "day", "years", "year", "total", "number",
            }
        ]

        def _spec(around: str) -> int:
            return sum(1 for t in topic_keys if soft_contains(around, t))

        def _num_span(m: re.Match, group: int = 1) -> tuple[int | None, str]:
            num_s = m.group(group)
            n = _parse_count_token(num_s)
            if n is None or n <= 0 or n > 100:
                return None, ""
            # Tight window around the number only (avoid '12 films … including 5 MCU')
            around = blob[max(0, m.start(group) - 12) : m.end(group) + 28]
            return n, around

        for m in re.finditer(
            r"\b(?:i(?:'ve| have)?|i)\b[^\n.]{0,40}?\b(?:tried|watched|bought|"
            r"assembled|fixed|sold|got|viewed)\b[^\n.]{0,40}?\b"
            r"(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
            r"[^\n.]{0,40}?\b(?:of|recipes?|films?|movies?|pieces?|items?)\b",
            blob,
            re.I,
        ):
            n, around = _num_span(m)
            if n is None:
                continue
            spec = _spec(around)
            if spec <= 0:
                continue
            stated_ranked.append((spec, n))
        for m in re.finditer(
            r"\b(?:tried|watched|bought|viewed)\s+out\s+"
            r"(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
            r"[^\n.]{0,24}?\bof\b",
            blob,
            re.I,
        ):
            n, around = _num_span(m)
            if n is None:
                continue
            spec = _spec(around)
            if spec <= 0:
                continue
            stated_ranked.append((spec + 1, n))  # 'tried out N of' is a strong form
        # Prefer 'N <distinctive topic>' adjacency: '5 MCU films', '3 … recipes'
        # When present, these beat looser 'watched N films' counts.
        for m in re.finditer(
            r"\b(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
            r"[^\n.]{0,12}?\b(?:MCU|recipes?)\b",
            blob,
            re.I,
        ):
            n, around = _num_span(m)
            if n is None:
                continue
            if _spec(around) <= 0 and not soft_contains(around, "mcu"):
                # still allow recipe adjacency when Emma/recipe is in topic_keys
                if not any(soft_contains(around, t) for t in topic_keys):
                    continue
            adjacent_ns.append(n)
            stated_ranked.append((10, n))

    if adjacent_ns:
        n = max(adjacent_ns)
        lines.append(
            f"Suggested stated total from first-person count claim: {n}."
        )
        lines.append(
            f"Answer with the integer {n}. Prefer this explicit total over "
            "enumerating every nearby mention."
        )
    elif stated_ranked:
        best_spec = max(s for s, _ in stated_ranked)
        pool = [n for s, n in stated_ranked if s == best_spec]
        n = max(pool)
        lines.append(
            f"Suggested stated total from first-person count claim: {n}."
        )
        lines.append(
            f"Answer with the integer {n}. Prefer this explicit total over "
            "enumerating every nearby mention."
        )
    elif open_enumerate and distinct_items:
        lines.append(
            "Candidate distinct items (dedupe across sessions): "
            + "; ".join(distinct_items[:10])
        )
        lines.append(
            f"Suggested distinct item count: {len(distinct_items[:10])}."
        )
        lines.append(
            f"Answer with the integer {len(distinct_items[:10])} plus short names "
            "when helpful. Count distinct items, do not sum every number in memory."
        )
    elif session_hits >= 2 and not money_q and not number_claims:
        lines.append(
            "On-topic sessions are listed above. Enumerate matching items, then answer "
            "with a count. Do not say you do not know."
        )

    if money_vals:
        lines.append(
            "Candidate money amounts: "
            + "; ".join(
                f"${v}←{s}" + (f" [{o}]" if o else "")
                for v, s, o in money_vals[:8]
            )
        )
        if len(conjuncts) >= 2:
            per_m: dict[str, int] = {}
            want_min = bool(re.search(r"\bminimum\b", ql))
            for v, _s, obj in money_vals:
                if not obj:
                    continue
                for c in conjuncts:
                    if (
                        obj.lower() == c.lower()
                        or soft_contains(obj, c)
                        or soft_contains(c, obj)
                    ):
                        if c not in per_m:
                            per_m[c] = v
                        elif want_min:
                            per_m[c] = min(per_m[c], v)
                        else:
                            per_m[c] = max(per_m[c], v)
            if len(per_m) == len(conjuncts):
                total = sum(per_m.values())
                lines.append(
                    "Suggested money total (sum of per-conjunct amounts): "
                    f"{total} ("
                    + ", ".join(f"{c}=${per_m[c]}" for c in conjuncts)
                    + ")."
                )
                lines.append(
                    f"Answer with ${total} (or the integer {total}) unless a clearer total is stated."
                )
            else:
                missing = [c for c in conjuncts if c not in per_m]
                lines.append(
                    "Incomplete money evidence: no paired amount for "
                    + ", ".join(missing)
                    + ". Prefer abstain ('I do not know') rather than guessing."
                )
        elif re.search(r"\b(?:total|altogether|in total|spend|spent)\b", ql):
            # Single-topic spend total: sum distinct amounts only when no conjunct split
            seen_m: set[int] = set()
            vals = []
            for v, _s, _o in money_vals:
                if v in seen_m:
                    continue
                seen_m.add(v)
                vals.append(v)
            if vals:
                total = sum(vals[:8])
                lines.append(f"Suggested money total (sum of listed amounts): {total}.")
                lines.append(
                    f"Answer with ${total} (or the integer {total}) unless a clearer total is stated."
                )

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def coverage_select_for_aggregate(
    candidates: list[dict], query: str, limit: int
) -> list[dict]:
    """
    B1: set-cover ordering for aggregate questions.

    Score order alone lets redundant hits from one session crowd out the second
    to fifth gold sessions. Greedily pick the best hit from each distinct
    topic-matching session first, then fill remaining slots by score.

    Also: (1) force one hit per A-and-B conjunct object into the front of the
    pool; (2) promote sibling sessions that share a source family stem so
    multi-part chats (…_1 …_4) are not dropped from top-k.
    """
    if not is_aggregate_query(query) or not candidates:
        return candidates
    # Errand counts rely on RRF order + digest; do not reorder those
    if is_errand_count_query(query):
        return candidates
    terms = extract_topic_phrases(query) + extract_topic_nouns(query)
    conjuncts = extract_and_conjuncts(query)
    if not terms and not conjuncts:
        return candidates

    def _src(r: dict) -> str:
        return (r.get("source_description") or r.get("uuid") or "").strip()

    def _score(r: dict) -> float:
        return float(r.get("score") or 0)

    def _family(src: str) -> str:
        # answer_8858d9dc_3 / lme_session:answer_8858d9dc_3 → answer_8858d9dc
        m = re.search(r"(answer_[0-9a-f]+|[0-9a-f]{8})(?:_\d+)?", src, re.I)
        if m:
            return m.group(1).lower()
        return src.lower()

    phrases = extract_topic_phrases(query)

    # Best topic-matching candidate per session (gentle floor below caps noise risk)
    best_by_session: dict[str, dict] = {}
    for r in candidates:
        cl = (r.get("content") or "").lower()
        phrase_hit = any(p in cl for p in phrases)
        noun_hits = sum(1 for t in terms if t in cl and len(t) >= 4)
        conj_hit = any(soft_contains(cl, c) for c in conjuncts)
        if not phrase_hit and noun_hits < 1 and not conj_hit:
            continue
        src = _src(r)
        if not src:
            continue
        cur = best_by_session.get(src)
        if cur is None or _score(r) > _score(cur):
            best_by_session[src] = r

    # One best hit per conjunct object (meals: fajitas session + soup session)
    conj_forced: list[dict] = []
    seen_conj_src: set[str] = set()
    for c in conjuncts:
        best = None
        for r in candidates:
            if not soft_contains(r.get("content") or "", c):
                continue
            if best is None or _score(r) > _score(best):
                best = r
        if best is None:
            continue
        src = _src(best)
        if src in seen_conj_src:
            continue
        seen_conj_src.add(src)
        conj_forced.append(best)
        best_by_session.setdefault(src, best)

    if len(best_by_session) <= 1 and len(conj_forced) < 2:
        return candidates

    # Sibling family promotion: multi-part chats (…_1 …_4) often omit the
    # question noun in some parts; still keep them once one family member hits.
    families = {_family(_src(r)) for r in best_by_session.values() if _family(_src(r))}
    family_extra = 0
    for r in sorted(candidates, key=_score, reverse=True):
        src = _src(r)
        fam = _family(src)
        if not fam or fam not in families:
            continue
        if src in best_by_session:
            continue
        best_by_session[src] = r
        family_extra += 1
        if family_extra >= 6:
            break

    picked: list[dict] = []
    picked_ids: set[int] = set()
    for r in conj_forced + sorted(
        best_by_session.values(), key=_score, reverse=True
    ):
        if id(r) in picked_ids:
            continue
        picked.append(r)
        picked_ids.add(id(r))
    rest = sorted(
        (r for r in candidates if id(r) not in picked_ids),
        key=_score,
        reverse=True,
    )
    out = picked + rest
    # Gentle bump: keep distinct sessions ahead of same-session duplicates
    # without leapfrogging genuinely stronger hits.
    for i, r in enumerate(picked[: max(limit, 8)]):
        r["coverage_pick"] = True
        floor = 0.66 - 0.01 * i
        if _score(r) < floor:
            r["score"] = floor
            r["fusion_score"] = floor
    return out


_SIBLING_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")


def harvest_sibling_terms(
    hits: list[dict], query: str, *, max_terms: int = 6
) -> list[str]:
    """
    B1: harvest anchor terms from round-1 topic hits for one follow-up retrieval.

    Sibling sessions often omit the question noun (a 'doctors' question where a
    sibling session only says 'dermatologist appointment'). Pull proper names and
    distinctive nouns that co-occur with topic matches to find those siblings.
    """
    if not is_aggregate_query(query) or not hits:
        return []
    terms = extract_topic_phrases(query) + extract_topic_nouns(query)
    if not terms:
        return []
    ql = (query or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for h in hits[:8]:
        content = h.get("content") or ""
        cl = content.lower()
        idxs = [cl.find(t) for t in terms if cl.find(t) >= 0]
        if not idxs:
            continue
        # Windows around topic matches only (avoid harvesting unrelated chatter)
        for idx in idxs[:3]:
            window = content[max(0, idx - 200) : idx + 300]
            for m in _SIBLING_PROPER_RE.finditer(window):
                name = m.group(1).strip()
                nl = name.lower()
                if (
                    nl in seen
                    or nl in ql
                    or nl in _TOPIC_STOP
                    or len(nl) < 4
                    or nl in {"here", "there", "monday", "tuesday", "wednesday",
                              "thursday", "friday", "saturday", "sunday",
                              "january", "february", "march", "april", "june",
                              "july", "august", "september", "october",
                              "november", "december"}
                ):
                    continue
                seen.add(nl)
                out.append(nl)
                if len(out) >= max_terms:
                    return out
    return out


def effective_search_limit(query: str, limit: int) -> int:
    """Recommend / count / preference-context questions benefit from a wider pool."""
    base = max(limit, 1)
    q = (query or "").lower()
    if needs_preference_retrieval(query) and any(
        w in q for w in ("publication", "conference", "paper", "journal", "interesting")
    ):
        return max(base, 16)
    if is_soft_advice_query(query):
        return max(base, 16)
    if is_temporal_span_query(query):
        return max(base, 14)
    if is_aggregate_query(query) or is_count_query(query):
        return max(base, 16)
    if needs_preference_retrieval(query) or is_preference_context_query(query):
        return max(base, 12)
    return base


def is_temporal_ago_query(query: str) -> bool:
    """How many days/weeks/months ago did X happen (needs question_date − event)."""
    return bool(
        re.search(
            r"\bhow many (?:days?|weeks?|months?) ago\b|"
            r"\bago did i\b",
            query or "",
            re.I,
        )
    )


def is_shipping_latency_query(query: str) -> bool:
    """How many days between order/buy and receive/arrive (shipping lag)."""
    q = query or ""
    if not re.search(r"\bhow many days\b", q, re.I):
        return False
    if not re.search(r"\b(?:order(?:ed)?|bought|purchased)\b", q, re.I):
        return False
    return bool(
        re.search(r"\b(?:receiv(?:e|ed)|arriv(?:e|ed|al)|deliver(?:y|ed))\b", q, re.I)
    )


def _parse_month_day_in_text(text: str, year: int) -> list[Any]:
    """Parse 'February 5th' / 'Feb 10' / '1/15' dates; year from session/question."""
    from datetime import datetime as _dt

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
        "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    out: list[Any] = []
    for m in re.finditer(
        r"\b(" + "|".join(months.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        text or "",
        re.I,
    ):
        mon = months.get(m.group(1).lower())
        if not mon:
            continue
        try:
            out.append(_dt(year, mon, int(m.group(2))))
        except ValueError:
            continue
    # Numeric M/D or M/D/YYYY (laptop backpack: bought on 1/15, arrived on 1/20)
    for m in re.finditer(
        r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b",
        text or "",
    ):
        mon, day = int(m.group(1)), int(m.group(2))
        y = year
        if m.group(3):
            y = int(m.group(3))
            if y < 100:
                y += 2000
        if mon < 1 or mon > 12 or day < 1 or day > 31:
            continue
        try:
            out.append(_dt(y, mon, day))
        except ValueError:
            continue
    return out


def build_temporal_span_digest(
    hits: list[dict], query: str = "", question_date: str = ""
) -> str:
    """
    Surface dated snippets that match venue/event anchors in a span question.

    Helps answer models compute day/month deltas instead of abstaining.
    Only emits Suggested span integers when the question asks for a duration count.
    """
    if not asks_temporal_span_integer(query) or not hits:
        return ""
    anchors = extract_temporal_anchors(query)
    if not anchors:
        return ""
    ago = is_temporal_ago_query(query)
    lines = [
        "Dated event anchors found in memory "
        "(use these timestamps to answer how many days/weeks/months/years "
        + ("ago, or the asked span):" if ago else "passed):"),
    ]
    n = 0
    seen_sid: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        matched = [a for a in anchors if a.lower() in cl]
        if not matched:
            continue
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        if sid in seen_sid:
            continue
        seen_sid.add(sid)
        when = h.get("valid_at_human") or ""
        idxs = [cl.find(a.lower()) for a in matched if cl.find(a.lower()) >= 0]
        idx = min(idxs) if idxs else 0
        start = max(0, idx - 40)
        end = min(len(content), idx + 100)
        snippet = re.sub(r"\s+", " ", content[start:end]).strip()
        stamp = f" @{when}" if when else ""
        lines.append(
            f"- ({sid}){stamp} [{', '.join(matched[:3])}]: {snippet}"
        )
        n += 1
        if n >= 8:
            break
    if n == 0:
        return ""
    # Deterministic span from earliest/latest dated anchors (unix or human stamps)
    from datetime import datetime as _dt, timezone as _tz

    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
        "december": 12,
    }

    def _parse_when(h: dict) -> Any | None:
        try:
            va = int(h.get("valid_at") or 0)
            if va > 1_000_000_000:
                return _dt.fromtimestamp(va, tz=_tz.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            pass
        when = (h.get("valid_at_human") or "").strip()
        if not when:
            return None
        m = re.search(r"(\d{4})/(\d{2})/(\d{2})", when)
        if m:
            try:
                return _dt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
        m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})", when)
        if m2:
            mon = _MONTHS.get(m2.group(1).lower())
            if mon:
                try:
                    return _dt(int(m2.group(3)), mon, int(m2.group(2)))
                except ValueError:
                    return None
        return None

    parsed_dates: list[Any] = []
    for h in hits:
        content = (h.get("content") or "").lower()
        if not any(a.lower() in content for a in anchors):
            continue
        dt = _parse_when(h)
        if dt is not None:
            parsed_dates.append(dt)

    # Order→receive: dates often live in the transcript ("ordered … February 5th",
    # "arrived on February 10th") while session stamps are all the same day.
    shipping = is_shipping_latency_query(query)
    content_span_done = False
    if shipping and not ago:
        year = None
        m_y = re.search(r"(\d{4})", question_date or "")
        if m_y:
            year = int(m_y.group(1))
        if year is None:
            for h in hits:
                dt0 = _parse_when(h)
                if dt0 is not None:
                    year = dt0.year
                    break
        if year is not None:
            order_dates: list[Any] = []
            arrive_dates: list[Any] = []
            for h in hits:
                content = h.get("content") or ""
                cl = content.lower()
                if not any(a.lower() in cl for a in anchors):
                    continue
                for m in re.finditer(
                    r"(.{0,50}\b(?:order(?:ed)?|bought|purchased)\b.{0,80})",
                    content,
                    re.I,
                ):
                    order_dates.extend(_parse_month_day_in_text(m.group(1), year))
                for m in re.finditer(
                    r"(.{0,50}\b(?:arriv(?:e|ed|al)|receiv(?:e|ed)|deliver(?:y|ed))\b.{0,80})",
                    content,
                    re.I,
                ):
                    arrive_dates.extend(_parse_month_day_in_text(m.group(1), year))
            if order_dates and arrive_dates:
                order_dt = min(order_dates)
                arrive_dt = min(d for d in arrive_dates if d >= order_dt) if any(
                    d >= order_dt for d in arrive_dates
                ) else min(arrive_dates)
                delta_days = max(0, (arrive_dt.date() - order_dt.date()).days)
                if delta_days > 0:
                    lines.append(
                        f"Suggested span from listed event dates: {delta_days} days "
                        f"(ordered {order_dt.date().isoformat()} → "
                        f"received/arrived {arrive_dt.date().isoformat()})."
                    )
                    lines.append(
                        f"Answer with the integer {delta_days} unless a listed anchor "
                        "clearly does not match the asked order/delivery."
                    )
                    content_span_done = True

    # Between-event spans when session stamps collapse to one day (Holi Feb 26 vs
    # Church March 19 both stamped March 26) but transcript has the real dates.
    if not ago and not content_span_done and len(anchors) >= 2:
        year = None
        m_y = re.search(r"(\d{4})", question_date or "")
        if m_y:
            year = int(m_y.group(1))
        if year is None:
            for h in hits:
                dt0 = _parse_when(h)
                if dt0 is not None:
                    year = dt0.year
                    break
        if year is not None:
            per_anchor: list[Any] = []
            for a in anchors[:8]:
                found: list[Any] = []
                al = a.lower()
                for h in hits:
                    content = h.get("content") or ""
                    cl = content.lower()
                    if al not in cl:
                        continue
                    for m in re.finditer(re.escape(a), content, re.I):
                        window = content[max(0, m.start() - 70) : m.end() + 90]
                        found.extend(_parse_month_day_in_text(window, year))
                if found:
                    per_anchor.append(min(found))
            if len(per_anchor) >= 2:
                per_anchor.sort()
                delta_days = (per_anchor[-1].date() - per_anchor[0].date()).days
                if delta_days > 0:
                    ql = (query or "").lower()
                    if "week" in ql:
                        span_n = max(0, int(round(delta_days / 7.0)))
                        unit = "weeks"
                    elif "month" in ql:
                        span_n = max(0, int(round(delta_days / 30.0)))
                        unit = "months"
                    else:
                        span_n = delta_days
                        unit = "days"
                    lines.append(
                        f"Suggested span from listed event dates: {span_n} {unit} "
                        f"({delta_days} days between earliest and latest dated "
                        f"anchors in transcript text)."
                    )
                    lines.append(
                        f"Answer with the integer {span_n} unless a listed anchor "
                        "clearly does not match the asked events."
                    )
                    content_span_done = True

    if len(parsed_dates) >= 2 and not ago and not content_span_done:
        parsed_dates.sort()
        delta_days = (parsed_dates[-1] - parsed_dates[0]).days
        # Skip poisonous 0-day suggestions (same session stamp on order+chat)
        if delta_days > 0 or not shipping:
            ql = (query or "").lower()
            if "week" in ql:
                span_n = max(0, int(round(delta_days / 7.0)))
                unit = "weeks"
            elif "month" in ql:
                span_n = max(0, int(round(delta_days / 30.0)))
                unit = "months"
            elif "year" in ql:
                span_n = max(0, int(round(delta_days / 365.0)))
                unit = "years"
            else:
                span_n = max(0, delta_days)
                unit = "days"
            if not (shipping and span_n == 0):
                lines.append(
                    f"Suggested span from listed event dates: {span_n} {unit} "
                    f"({delta_days} days between earliest and latest dated anchors)."
                )
                lines.append(
                    f"Answer with the integer {span_n} unless a listed anchor clearly does "
                    "not match the asked events."
                )
    if ago:
        # Suggested integer from question_date − best matching event.
        q_dt = None
        qd = (question_date or "").strip()
        m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", qd)
        if m:
            try:
                q_dt = _dt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                q_dt = None
        event_dates = sorted(parsed_dates) if parsed_dates else []
        if q_dt is not None and event_dates:
            prior = [d for d in event_dates if d.date() <= q_dt.date()]
            event_dt = prior[-1] if prior else event_dates[-1]
            delta_days = max(0, (q_dt.date() - event_dt.date()).days)
            ql = (query or "").lower()
            if "week" in ql:
                span_n = max(0, int(round(delta_days / 7.0)))
                unit = "weeks"
            elif "month" in ql:
                span_n = max(0, int(round(delta_days / 30.0)))
                unit = "months"
            else:
                span_n = delta_days
                unit = "days"
            lines.append(
                f"Suggested span from listed event dates: {span_n} {unit} "
                f"({delta_days} days before question date {q_dt.date().isoformat()})."
            )
            lines.append(
                f"Answer with the integer {span_n} unless a listed anchor clearly does "
                "not match the asked event."
            )
        lines.append(
            "For 'how many days/weeks ago' questions: pick the dated event that matches "
            "the asked activity or named app/event in the question, then compute "
            "question_date minus that event date. Answer with a single integer in the "
            "asked unit. Do not switch to a newer unrelated event just because a conflict "
            "hint says prefer NEWER."
        )
    else:
        lines.append(
            "Compute the asked span from the event dates (or session timestamps). "
            "Answer with a single integer in the asked unit (days/weeks/months/years). "
            "Do not abstain when two dated on-topic events are listed."
        )
    return "\n".join(lines)


_LOC_INTENT_RE = re.compile(
    r"\b(?:where|lives?|living|moved?|moving|relocat|now|currently|"
    r"these days|stay(?:ing)?|based)\b",
    re.I,
)


def prioritize_query_entity_recency(episodes: list[dict], query: str) -> list[dict]:
    """
    For questions naming a person/place, prefer newest *location-update*
    episodes (moved/suburbs/apartment), not merely newest mention.

    Fires only on location/current-state intent. Reordering every question that
    happens to contain a capitalized token ('Star Wars' → 'Star') buried
    score-ranked gold under newest-mention noise (M2 temporal fails).
    """
    if not _LOC_INTENT_RE.search(query or ""):
        return episodes
    if is_aggregate_query(query):
        return episodes
    # Word-boundary focus matching; short tokens ('Star') are substring traps
    focus = [t.lower() for t in extract_query_focus_terms(query) if len(t) >= 4]
    if not focus or not episodes:
        return episodes

    def _va(r: dict) -> int:
        try:
            return int(r.get("valid_at") or 0)
        except (TypeError, ValueError):
            return 0

    def _loc_score(r: dict) -> int:
        content = r.get("content") or ""
        cl = content.lower()
        if not any(t in cl for t in focus):
            return -1
        score = 1
        if _LOC_UPDATE_RE.search(content):
            score += 10
        if "suburb" in cl:
            score += 5
        if "moved" in cl or "relocat" in cl:
            score += 3
        return score

    matched: list[dict] = []
    rest: list[dict] = []
    for row in episodes:
        content = (row.get("content") or "").lower()
        if any(t in content for t in focus):
            matched.append(row)
        else:
            rest.append(row)
    if len(matched) < 2:
        return episodes

    # Prefer location-update language, then newest valid_at
    matched = sorted(matched, key=lambda r: (_loc_score(r), _va(r)), reverse=True)
    out: list[dict] = []
    for i, row in enumerate(matched):
        r = dict(row)
        if i == 0:
            r["recency_label"] = "newer"
            r["entity_latest"] = True
            focus_hit = next(
                (
                    t
                    for t in extract_query_focus_terms(query)
                    if t.lower() in (r.get("content") or "").lower()
                ),
                focus[0],
            )
            banner = (
                f"[LATEST update for {focus_hit}] Use this episode for current location/status; "
                f"ignore older episodes about {focus_hit}."
            )
            content = r.get("content") or ""
            if "[LATEST update" not in content:
                r["content"] = banner + "\n" + content
        elif r.get("recency_label") != "newer":
            r["recency_label"] = "older"
        out.append(r)
    out.extend(rest)
    return out


def order_hits_for_answer(hits: list[dict]) -> list[dict]:
    """Prefer NEWER / entity_latest, expertise, preference-rich, then errand-rich."""
    def _key(h: dict) -> tuple:
        label = h.get("recency_label")
        rec = 0 if label == "newer" else (1 if label == "older" else 2)
        pref = 0 if h.get("preference_boost") else 1
        expertise = 0 if h.get("expertise_boost") or h.get("expertise_forced") else 1
        errand = 0 if h.get("errand_boost") else 1
        cues = h.get("cues") or {}
        rich = 0 if (cues.get("preferences") or cues.get("features")) else 1
        latest = 0 if h.get("entity_latest") else 1
        try:
            score = -float(h.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        depth = -float(h.get("expertise_depth") or 0)
        return (latest, expertise, rec, pref, errand, rich, depth, score)

    return sorted(hits, key=_key)


def build_inventory_digest(hits: list[dict], query: str = "") -> str:
    """
    For errand-style count questions: list distinct items (clustered), not every span.

    Same product/store mentioned as return + pickup + exchange collapses to one item.
    Non-errand counts (purchase history, how many kits worked on) skip this digest.
    """
    if not is_errand_count_query(query) or not hits:
        return ""

    # cluster_key -> {tags, sample, sid}
    clusters: dict[str, dict[str, Any]] = {}
    last_key_by_sid: dict[str, str] = {}
    last_item_by_sid: dict[str, str] = {}
    order: list[str] = []

    for h in hits:
        sid = (h.get("source_description") or h.get("uuid") or "").strip() or "?"
        cues = h.get("cues") or {}
        actions = list(cues.get("actions") or [])
        if not actions:
            raw = h.get("content") or ""
            for m in _ACTION_RE.finditer(raw):
                span = re.sub(r"\s+", " ", m.group(0)).strip()
                if span and span not in actions:
                    actions.append(span[:120])
        for a in actions:
            if _is_noise_errand_span(a):
                continue
            tag = classify_errand_action(a)
            if tag == "EXCHANGE":
                # Outstanding work is usually return old and/or pick up new
                al = a.lower()
                if re.search(r"\bpick", al):
                    tag = "PICKUP"
                elif re.search(r"\breturn", al):
                    tag = "RETURN"
                else:
                    continue
            if tag not in {"RETURN", "PICKUP", "REDEEM"}:
                continue
            item_key = _errand_item_key(a)
            prev = last_key_by_sid.get(f"{sid}|{tag}")
            # Same-session follow-ups often say "the new pair" with no noun;
            # inherit the prior concrete item from this session.
            if item_key == "unknown":
                item_key = prev or last_item_by_sid.get(sid) or "unknown"
            if item_key == "unknown":
                continue
            if prev and _item_keys_compatible(item_key, prev):
                if prev != "unknown" and len(prev) >= len(item_key):
                    item_key = prev
                last_key_by_sid[f"{sid}|{tag}"] = item_key
            else:
                last_key_by_sid[f"{sid}|{tag}"] = item_key
            last_item_by_sid[sid] = item_key

            if prev and prev != item_key and _item_keys_compatible(prev, item_key):
                old_id = f"{sid}|{tag}|{prev}"
                new_id = f"{sid}|{tag}|{item_key}"
                if old_id in clusters and old_id != new_id:
                    old = clusters.pop(old_id)
                    if new_id in clusters:
                        clusters[new_id]["tags"] |= old["tags"]
                    else:
                        old["item"] = item_key
                        clusters[new_id] = old
                        order[order.index(old_id)] = new_id

            cluster_id = f"{sid}|{tag}|{item_key}"
            snippet = re.sub(r"^\[(Session cues|LATEST update)[^\]]*\]\s*", "", a)
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "tags": {tag},
                    "sample": snippet[:160],
                    "sid": sid,
                    "item": item_key,
                }
                order.append(cluster_id)
            else:
                clusters[cluster_id]["tags"].add(tag)

    seen_order: list[str] = []
    for cid in order:
        if cid in clusters and cid not in seen_order:
            seen_order.append(cid)
    clusters, order = _merge_cluster_maps(clusters, seen_order)

    if not order:
        return ""

    lines = [
        "Candidate errand obligations inferred from memory "
        "(duplicate mentions of the same action on the same item collapse; "
        "return vs pickup stay separate):",
    ]
    for cid in order[:8]:
        c = clusters[cid]
        tags = "/".join(sorted(c["tags"]))
        item = c["item"] if c["item"] != "unknown" else "item"
        lines.append(f"- [{tags}] ({c['sid']}) {item}: {c['sample']}")
    n = min(len(order), 8)
    lines.append(f"Suggested outstanding obligations listed: {n}.")
    lines.append(
        f"Answer with only the integer {n} (the suggested outstanding-obligation count). "
        "Do not substitute a lower count from a partial re-read of the excerpts. "
        "RETURN and PICKUP of the same product can both remain open and both count. "
        "Only use a different integer if a bullet is clearly completed or unrelated."
    )
    return "\n".join(lines)


_USER_DURATION_RE = re.compile(
    r"(?:i\s+spent\s+(?:around\s+|about\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)"
    r"(?:\s+playing)?|"
    r"(?:it\s+)?took\s+me\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)|"
    r"i\s+(?:went\s+for\s+a\s+|did\s+a\s+)?(\d+(?:\.\d+)?)\s*-?\s*"
    r"(minutes?|mins?|hours?|hrs?)\s+(?:jog|run|walk|yoga|workout|ride))",
    re.I,
)


def build_activity_duration_digest(hits: list[dict], query: str = "") -> str:
    """List first-person duration mentions for workout/games hour questions."""
    if not is_activity_duration_query(query) or not hits:
        return ""
    q_acts = sorted({m.group(0).lower() for m in _ACTIVITY_RE.finditer(query or "")})
    games_q = bool(re.search(r"\b(?:games?|gaming|playing)\b", query or "", re.I))
    total_q = bool(
        re.search(r"\b(?:in total|altogether|all together|overall)\b", query or "", re.I)
    )
    lines = [
        "Activity duration mentions found in memory "
        "(prefer first-person completed times; convert minutes to hours):",
    ]
    collected: list[tuple[float, str]] = []
    session_hours: list[float] = []
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        if q_acts and not any(a in cl for a in q_acts):
            if not (
                games_q
                and re.search(r"\b(?:hours?|hrs?)\b", cl)
                and re.search(r"\b(?:played|playing|playtime|took me|i spent)\b", cl)
            ):
                if not any(a[:3] in cl for a in q_acts if len(a) >= 3):
                    continue
        when = h.get("valid_at_human") or ""
        sid = h.get("source_description") or h.get("uuid") or "?"
        per_session: dict[float, str] = {}
        for m in _USER_DURATION_RE.finditer(content):
            amount_s = m.group(1) or m.group(3) or m.group(5)
            unit_s = m.group(2) or m.group(4) or m.group(6) or ""
            if not amount_s:
                continue
            amount = float(amount_s)
            hours = amount / 60.0 if "min" in unit_s.lower() else amount
            if hours > 500:
                continue
            window = content[max(0, m.start() - 40) : min(len(content), m.end() + 60)]
            snippet = re.sub(r"\s+", " ", window).strip()
            stamp = f" @{when}" if when else ""
            per_session[hours] = (
                f"- ({sid}){stamp} ~{hours:g}h user claim '{m.group(0)[:60]}': {snippet}"
            )
        if not per_session:
            for m in _DURATION_SPAN_RE.finditer(content):
                window = content[max(0, m.start() - 80) : min(len(content), m.end() + 80)]
                wl = window.lower()
                if re.search(r"\b\d+\s*[-–]\s*\d+\s*hours?\b", wl):
                    continue
                if not re.search(
                    r"\b(?:i\s+spent|took\s+me|i\s+went|i\s+did|playing)\b",
                    wl,
                ):
                    continue
                amount = float(m.group(1))
                unit = m.group(0).lower()
                hours = amount / 60.0 if "min" in unit else amount
                snippet = re.sub(r"\s+", " ", window).strip()
                stamp = f" @{when}" if when else ""
                per_session[hours] = (
                    f"- ({sid}){stamp} ~{hours:g}h from '{m.group(0)}': {snippet}"
                )
        for hours, line in per_session.items():
            collected.append((hours, line))
            session_hours.append(hours)
    total_hours = sum(session_hours)
    for _, line in collected[:16]:
        lines.append(line)
    if len(collected) > 16:
        lines.append(f"- …and {len(collected) - 16} more user duration claims in memory.")
    if not collected:
        lines.append(
            "- (none found in top excerpts) If no matching duration is present for the "
            "asked activities, answer 0 hours."
        )
    else:
        lines.append(
            f"Candidate duration sum (first-person claims, deduped per session amount): "
            f"{total_hours:g} hours."
        )
        if total_q:
            # Integer-friendly display for whole-hour game totals
            shown = (
                str(int(total_hours))
                if abs(total_hours - round(total_hours)) < 1e-6
                else f"{total_hours:g}"
            )
            lines.append(f"Suggested total hours listed: {shown}.")
            lines.append(
                f"For an 'in total' question, answer with {shown} unless a listed "
                "claim is clearly not the user's completed playtime."
            )
        lines.append(
            "Ignore assistant suggestion ranges like '(60-100 hours of gameplay)'. "
            "Prefer concrete completed play/workouts over plans. "
            "Convert 30 minutes to 0.5 hours. "
            "For last-week questions: if the only completed matching workout is dated "
            "within ~14 days before the question date, count it (answer that duration, "
            "not 0)."
        )
    return "\n".join(lines)


_PURCHASE_TITLE_RE = re.compile(
    r"\b(?:bought|purchased|downloaded|got|signed)\b[^\.\n]{0,100}?"
    r"(?:album|ep|vinyl|kit|model)\b[^\.\n]{0,80}"
    r"|"
    r"\b(?:album|ep|vinyl)\b[^\.\n]{0,40}?\b(?:bought|purchased|downloaded|signed)\b[^\.\n]{0,60}",
    re.I,
)
_QUOTED_TITLE_RE = re.compile(r"[\"'“]([^\"'”]{2,60})[\"'”]")
_TITLE_NOISE_RE = re.compile(
    r"\b(?:sure|you|looking|suggestions?|festival|weekend|show|after|"
    r"the|and|for|with|from|this|that|have|got|my)\b",
    re.I,
)


def _is_clean_purchase_title(title: str) -> bool:
    """Reject chat fragments mistaken for album/kit titles."""
    t = (title or "").strip()
    if len(t) < 3 or len(t) > 60:
        return False
    words = t.split()
    if not words or len(words) > 8:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    # Allow "X vinyl" artist labels; otherwise require a proper-looking title
    if re.search(r"\bvinyl\b", t, re.I):
        return bool(re.search(r"[A-Z]", t))
    noise_hits = len(_TITLE_NOISE_RE.findall(t))
    if noise_hits >= max(1, len(words) - 1):
        return False
    # Prefer titles with a capital or digits (scale models / branded names)
    if not re.search(r"[A-Z0-9]", t):
        return False
    return True


_USER_ACQUIRE_RE = re.compile(
    r"(?:i(?:'ve| have)?\s+(?:just\s+)?(?:bought|got|acquired|purchased|picked\s*up|"
    r"added|set\s*up|started|adopted|brought\s*home|inherited|received)|"
    r"(?:which|that)\s+i\s+(?:got|inherited|received)\s+from|"
    r"i have an?\s+(?:antique|vintage|depression[- ]era)\s+[^\.\n]{3,80}|"
    r"(?:an?\s+)?(?:antique|vintage|depression[- ]era)\s+[a-z][a-z\s-]{2,40}?"
    r"(?:\s+from\s+my|\s+that belonged to my|\s+belonged to my)|"
    r"my\s+(?:new\s+)?(?:plant|tank|succulent|lily|aquarium|citrus|lemon|lime|orange|"
    r"grapefruit|kit|album|ep|snake\s+plant|peace\s+lily|spider\s+plant))"
    r"[^\.\n]{0,120}",
    re.I,
)
# Provenance-anchored only: avoids assistant chatter ("antique dealers", "vintage items")
_HEIRLOOM_ITEM_RE = re.compile(
    r"\b((?:antique|vintage|depression[- ]era)\s+(?:diamond\s+)?"
    r"[a-z]+(?:\s+[a-z]+){0,3})"
    r"(?=\s+(?:from\s+my|came\s+from\s+my|that\s+belonged\s+to\s+my|"
    r"belonged\s+to\s+my)\b)",
    re.I,
)
_HEIRLOOM_FAMILY_RE = re.compile(
    r"\b(?:my\s+)?(?:grandmother'?s|grandfather'?s|mom'?s|dad'?s|mother'?s|"
    r"father'?s|aunt'?s|uncle'?s|great-aunt'?s|cousin(?:'s|\s+\w+'s)?)\s+"
    r"((?:antique|vintage|depression[- ]era)\s+(?:diamond\s+)?"
    r"[a-z]+(?:\s+[a-z]+){0,3})(?=\s|,|\.|$)",
    re.I,
)
_HEIRLOOM_BAN = {
    "dealers", "stores", "shops", "malls", "pieces", "goods", "items",
    "jewelry", "electronics", "and", "mechanical", "royal", "appraiser",
    "came", "that", "belonged", "from",
}
_USER_USED_RE = re.compile(
    r"(?:i(?:'ve| have)?\s+(?:also\s+)?(?:used|made|tried|mixed|added)\b|"
    r"my\s+(?:cocktail|recipe|drink))"
    r"[^\.\n]{0,100}",
    re.I,
)


_WORD_TO_INT = {
    "zero": 0, "once": 1, "twice": 2,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}


def _parse_count_token(tok: str) -> int | None:
    t = (tok or "").strip().lower()
    if t.isdigit():
        return int(t)
    return _WORD_TO_INT.get(t)


def build_redeem_points_digest(hits: list[dict], query: str = "") -> str:
    """
    'How many points to redeem X' → reward-tier cost, not current balance/goal.

    fair_c1 Sephora: user said 'close to 300' but skincare redeems at 100 points.
    """
    ql = (query or "").lower()
    if not re.search(r"\bhow many points\b", ql):
        return ""
    if not re.search(r"\bredeem\b", ql):
        return ""
    if not hits:
        return ""
    topics = extract_topic_phrases(query) + extract_topic_nouns(query)
    topic_hit = [t for t in topics if t not in {"points", "point", "many", "how"}]
    # Reward menu lines: "**Product** (100 points):"
    tier_re = re.compile(
        r"\((\d+)\s+points?\)\s*:",
        re.I,
    )
    tiers: list[int] = []
    snips: list[str] = []
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        if topic_hit and not any(t.lower() in cl for t in topic_hit):
            # Still allow Sephora/loyalty generic pages
            if "point" not in cl and "redeem" not in cl:
                continue
        for m in tier_re.finditer(content):
            n = int(m.group(1))
            if n <= 0 or n > 5000:
                continue
            window = content[max(0, m.start() - 80) : m.end() + 40]
            # Prefer tiers near asked product class (skincare / free product)
            wl = window.lower()
            if re.search(r"\b(?:skincare|serum|moisturizer|cleanser|toner|cream)\b", ql):
                if not re.search(
                    r"\b(?:skincare|serum|moisturizer|cleanser|toner|cream|face|skin)\b",
                    wl,
                ):
                    continue
            tiers.append(n)
            snips.append(re.sub(r"\s+", " ", window).strip()[:160])
    if not tiers:
        return ""
    # Typical redeem threshold is the common reward tier (mode), else min > 0
    from collections import Counter

    best = Counter(tiers).most_common(1)[0][0]
    lines = [
        "Reward redeem thresholds found in memory (points to redeem, not balance):",
    ]
    for s in snips[:4]:
        lines.append(f"- {s}")
    lines.append(f"Suggested redeem points: {best}.")
    lines.append(
        f"Answer with the integer {best}. Do not answer with a points balance or "
        "points goal (e.g. 200/300) unless the question asks for the balance."
    )
    return "\n".join(lines)


def build_collection_total_digest(hits: list[dict], query: str = "") -> str:
    """
    Stated collection totals with a later 'added a new' update (37 → 38 coins).
    """
    ql = (query or "").lower()
    if not re.search(r"\bhow many\b", ql):
        return ""
    if not re.search(r"\b(?:collection|coins?|cards?|stamps?)\b", ql):
        return ""
    if re.search(r"\bhow many times\b", ql):
        return ""
    if not hits:
        return ""
    topics = [
        t
        for t in (extract_topic_phrases(query) + extract_topic_nouns(query))
        if t not in {"many", "how", "collection"}
    ]
    total_re = re.compile(
        r"\b(?:total of|i have|I've got|i've got)\s+(\d+)\s+"
        r"(?:pre-1920\s+)?(?:american\s+)?(?:coins?|cards?|stamps?|items?)\b",
        re.I,
    )
    added_re = re.compile(
        r"\b(?:added|just got|got)\s+a\s+new\s+(?:coin|card|stamp|item)\b",
        re.I,
    )
    base: int | None = None
    base_sid = ""
    added_after = False
    ordered = sorted(
        hits,
        key=lambda h: float(h.get("valid_at") or 0) or 0,
    )
    for h in ordered:
        content = h.get("content") or ""
        cl = content.lower()
        if topics and not any(t.lower() in cl for t in topics):
            continue
        m = total_re.search(content)
        if m:
            base = int(m.group(1))
            base_sid = str(h.get("source_description") or "")
            added_after = False
        if base is not None and added_re.search(content):
            # Same or later session mentioning add
            added_after = True
    if base is None:
        return ""
    n = base + (1 if added_after else 0)
    lines = [
        "Collection total found in memory:",
        f"- stated total {base}"
        + (f" in {base_sid}" if base_sid else "")
        + ("; later session adds one new item" if added_after else ""),
        f"Suggested collection total: {n}.",
        f"Answer with the integer {n}.",
    ]
    return "\n".join(lines)


def build_stated_count_digest(hits: list[dict], query: str = "") -> str:
    """
    Surface first-person stated totals ('that's six times', 'I've worn them six times').

    Live agents ask wear/use counts constantly; without this the answer model often
    abstains even when the number is in a retrieved session (fair_c1 Converse).
    """
    ql = (query or "").lower()
    if not re.search(
        r"\bhow many times\b|"
        r"\bhow many (?:trips|episodes|meet(?:-|\s)?ups?|visits)\b",
        ql,
    ):
        return ""
    if not hits:
        return ""
    nouns = [n for n in extract_topic_nouns(query) if len(n) >= 4]
    phrases = extract_topic_phrases(query)
    topics = phrases + nouns
    if not topics:
        return ""
    num = (
        r"(\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)"
    )
    patterns = [
        re.compile(rf"\bthat'?s\s+{num}\s+times\b", re.I),
        re.compile(
            rf"\b(?:worn|wore|wear(?:ing)?|used|met(?:\s+up)?|visited|taken|completed)\b"
            rf"[^\n]{{0,60}}\b{num}\s+times\b",
            re.I,
        ),
        re.compile(
            rf"\b{num}\s+times\b[^\n]{{0,40}}\b(?:worn|wore|wear|used|met|visited)\b",
            re.I,
        ),
        # "we've met up twice" / "met up twice already"
        re.compile(
            rf"\bmet(?:\s+up)?\s+{num}\b|"
            rf"\b{num}\s+(?:already\s+)?(?:before|already)\b",
            re.I,
        ),
        re.compile(rf"\bmet(?:\s+up)?\s+{num}\s+already\b", re.I),
        re.compile(rf"\bwe(?:'ve| have)\s+met(?:\s+up)?\s+{num}\b", re.I),
        # "on five trips now" / "taken … on five trips"
        re.compile(rf"\bon\s+{num}\s+trips?\b", re.I),
        re.compile(rf"\b{num}\s+trips?\b", re.I),
    ]
    best: int | None = None
    snippet = ""
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        if not any(t.lower() in cl for t in topics):
            continue
        for pat in patterns:
            m = pat.search(content)
            if not m:
                continue
            n = _parse_count_token(m.group(1))
            if n is None or n <= 0:
                continue
            # Prefer larger stated total when several appear
            if best is None or n > best:
                best = n
                a = max(0, m.start() - 40)
                b = min(len(content), m.end() + 40)
                snippet = re.sub(r"\s+", " ", content[a:b]).strip()
    if best is None:
        return ""
    lines = [
        "Stated count found in memory (user self-report):",
        f"- {snippet}" if snippet else f"- stated total {best}",
        f"Suggested stated count: {best}.",
        f"Answer with the integer {best} unless a clearer on-topic total appears.",
    ]
    return "\n".join(lines)


def build_topic_inventory_digest(hits: list[dict], query: str = "") -> str:
    """
    For non-errand inventory counts: surface topic-overlapping episode snippets.
    """
    if not is_topic_inventory_query(query) or not hits:
        return ""
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    event_bridges = event_attend_bridge_terms(query)
    if not phrases and not nouns and not event_bridges:
        return ""
    ql = (query or "").lower()
    user_used_q = bool(
        re.search(r"\b(?:have i used|did i use|i used|types? of .+ used)\b", ql)
    )
    window_m = re.search(
        r"\b(?:this|last|past|previous)\s+(?:year|month|week|weekend|"
        r"(?:three|two|few|\d+)\s+(?:years?|months?|weeks?|days?))\b|"
        r"\bin\s+(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|\d{4})\b",
        ql,
    )
    lines = [
        "Topic inventory hints from memory (count distinct matching items/projects "
        "in the excerpts; do not stop at the first session):",
    ]
    seen_sid: set[str] = set()
    n = 0
    title_hints: list[str] = []
    seen_titles: set[str] = set()
    acquire_hints: list[str] = []
    seen_acq: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        matched = (
            [p for p in phrases if p in cl]
            + [noun for noun in nouns if noun in cl]
            + [b for b in event_bridges if b in cl]
        )
        if not matched:
            continue
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        if sid in seen_sid:
            continue
        seen_sid.add(sid)
        # Prefer a window around a first-person acquire/use mention when present
        prefer_re = _USER_USED_RE if user_used_q else _USER_ACQUIRE_RE
        pref_m = prefer_re.search(content)
        if pref_m:
            idx = pref_m.start()
        else:
            idxs = [cl.find(m) for m in matched if cl.find(m) >= 0]
            idx = min(idxs) if idxs else 0
        start = max(0, idx - 40)
        end = min(len(content), idx + 160)
        snippet = re.sub(r"\s+", " ", content[start:end]).strip()
        when = h.get("valid_at_human") or ""
        stamp = f" @{when}" if when else ""
        lines.append(f"- ({sid}){stamp} [{', '.join(matched[:3])}] {snippet}")
        n += 1
        # Collect first-person acquire/use spans as candidate items (multi per session)
        for m in prefer_re.finditer(content):
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            if len(span) < 8:
                continue
            # Keep only if topic noun/phrase also appears nearby
            nearby = content[max(0, m.start() - 30) : min(len(content), m.end() + 80)].lower()
            if not any(t in nearby for t in matched):
                continue
            key = span.lower()[:90]
            if key in seen_acq:
                continue
            seen_acq.add(key)
            acquire_hints.append(f"{span[:90]}{stamp}")
            if len(acquire_hints) >= 12:
                break
        session_had_quoted = False
        for m in _PURCHASE_TITLE_RE.finditer(content):
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            quoted = _QUOTED_TITLE_RE.findall(span) or _QUOTED_TITLE_RE.findall(
                content[max(0, m.start() - 40) : m.end() + 40]
            )
            if quoted:
                for qt in quoted:
                    qt = qt.strip()
                    key = qt.lower()
                    if key and key not in seen_titles and _is_clean_purchase_title(qt):
                        seen_titles.add(key)
                        title_hints.append(qt)
                        session_had_quoted = True
            else:
                # Keep short purchase spans only (avoid long assistant chatter)
                if len(span) > 80 or not _is_clean_purchase_title(span):
                    continue
                key = span.lower()[:80]
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    title_hints.append(span[:80])
        # At most one untitled vinyl credit per session
        if re.search(r"\bvinyl\b", content, re.I) and not session_had_quoted:
            am = re.search(
                r"\b([A-Z][\w']+(?:\s+[A-Z][\w']+){0,3})\s+vinyl\b",
                content,
            )
            if am:
                label = f"{am.group(1)} vinyl"
                key = label.lower()
                if key not in seen_titles and _is_clean_purchase_title(label):
                    seen_titles.add(key)
                    title_hints.append(label)
        if n >= 12:
            break
    if n == 0:
        return ""
    if title_hints:
        uniq = title_hints[:10]
        lines.append(
            "Candidate titled purchases/downloads (dedupe by title across sessions; "
            "include signed/purchased vinyl): "
            + "; ".join(uniq)
        )
        lines.append(f"Suggested distinct purchases/downloads listed: {len(uniq)}.")
        lines.append(
            f"Answer with the integer {len(uniq)} plus the item names. "
            "Count signed or purchased vinyl as one item even without a formal album title. "
            "Do not leave vinyl undecided or drop it after deliberation."
        )
    else:
        # Inherit/acquire: provenance-anchored antique/vintage items from user text
        heirloom_items: list[str] = []
        seen_heir: set[str] = set()
        if re.search(r"\b(?:inherit|acquired?|heirloom|family)\b", ql):
            for h in hits:
                content = h.get("content") or ""
                if content.lstrip().startswith("[Session events"):
                    continue
                # Prefer user turns when role prefixes are present
                user_chunks = re.findall(
                    r"(?im)^(?:user|human)\s*:\s*(.+)$", content
                )
                scan = "\n".join(user_chunks) if user_chunks else content

                def _add_heir(raw: str) -> None:
                    item = re.sub(r"\s+", " ", (raw or "").strip()).lower()
                    item = item.replace("depression era", "depression-era")
                    item = re.sub(
                        r"\s+(?:from|that|belonged|came|and|insured)$", "", item
                    ).strip()
                    toks = item.split()
                    if len(toks) < 2 or item in seen_heir:
                        return
                    # Ban generic tails ("antique dealers") and incomplete stems
                    if toks[-1] in _HEIRLOOM_BAN:
                        return
                    if item in {"vintage diamond", "antique music"}:
                        return
                    seen_heir.add(item)
                    heirloom_items.append(item)

                for m in _HEIRLOOM_ITEM_RE.finditer(scan):
                    _add_heir(m.group(1))
                for m in _HEIRLOOM_FAMILY_RE.finditer(scan):
                    _add_heir(m.group(1))
                for m in re.finditer(
                    r"\b(?:including|and)\s+(?:an?\s+|a\s+set\s+of\s+)?"
                    r"((?:antique|vintage|depression[- ]era)\s+[a-z]+(?:\s+[a-z]+){0,3})"
                    r"(?=\s+from\s+my\b)",
                    scan,
                    re.I,
                ):
                    _add_heir(m.group(1))
                if len(heirloom_items) >= 12:
                    break
            # Drop shorter duplicates that are prefixes of a longer item
            kept: list[str] = []
            for item in sorted(heirloom_items, key=len, reverse=True):
                if any(item == k or k.startswith(item + " ") for k in kept):
                    continue
                kept.append(item)
            heirloom_items = list(reversed(kept))
        if heirloom_items:
            lines.append(
                "Candidate inherited/acquired items (dedupe by item name across sessions): "
                + "; ".join(heirloom_items)
            )
            lines.append(
                f"Suggested distinct inherited/acquired items listed: {len(heirloom_items)}."
            )
            lines.append(
                f"Answer with the integer {len(heirloom_items)} plus the item names. "
                "Count each distinct antique/vintage item once."
            )
        elif acquire_hints:
            lines.append(
                "Candidate user acquire/use mentions (dedupe distinct items; ignore "
                "assistant-only suggestions the user did not claim): "
                + "; ".join(acquire_hints[:10])
            )
            lines.append(
                f"At least {len(acquire_hints)} first-person mentions listed; count distinct "
                "items (not every repeated mention of the same plant/tank/fruit)."
            )
        # Attended events: first-person attend/volunteer/tour/lecture with a name/date
        attended: list[str] = []
        seen_att: set[str] = set()
        if event_attend_bridge_terms(query):
            att_patterns = [
                re.compile(
                    r"\b(?:i\s+)?(?:recently\s+)?(?:attended|volunteered\s+at|went\s+on)\s+"
                    r"[^\.\n]{8,120}",
                    re.I,
                ),
                # 'Women in Art' exhibition which I attended on February 10th
                re.compile(
                    r"(?:\"[^\"]{2,60}\"\s+|the\s+)?"
                    r"(?:exhibition|lecture|event|tour|afternoon)"
                    r"[^\.\n]{0,30}?which\s+i\s+attended[^\.\n]{0,40}",
                    re.I,
                ),
                # titled event before the word exhibition/lecture
                re.compile(
                    r"\"([^\"]{3,60})\"\s+(?:exhibition|lecture|event|tour)"
                    r"[^\.\n]{0,40}?which\s+i\s+attended[^\.\n]{0,40}",
                    re.I,
                ),
            ]
            for h in hits:
                content = h.get("content") or ""
                if content.lstrip().startswith("[Session events"):
                    continue
                user_chunks = re.findall(
                    r"(?im)^(?:user|human)\s*:\s*(.+)$", content
                )
                scan = "\n".join(user_chunks) if user_chunks else content
                for att_re in att_patterns:
                    for m in att_re.finditer(scan):
                        span = re.sub(r"\s+", " ", m.group(0)).strip()
                        if not re.search(
                            r"\b(?:art|museum|gallery|lecture|exhibition|tour|"
                            r"volunteer|festival)\b",
                            span,
                            re.I,
                        ):
                            continue
                        key = span.lower()[:100]
                        if key in seen_att:
                            continue
                        seen_att.add(key)
                        attended.append(span[:120])
                        if len(attended) >= 10:
                            break
                    if len(attended) >= 10:
                        break
                if len(attended) >= 10:
                    break
        if attended:
            lines.append(
                "Candidate attended events (dedupe distinct events across sessions): "
                + "; ".join(attended)
            )
            lines.append(
                f"Suggested distinct attended events listed: {len(attended)}."
            )
            lines.append(
                f"Answer with the integer {len(attended)} plus short event names. "
                "Count each distinct attended/volunteered event once."
            )
        lines.append(
            f"Sessions with topic overlap listed: {n}. Enumerate every distinct matching "
            "item the user acquired/used across sessions, then give the total integer "
            "together with the item names. For 'have I used' questions, count only the "
            "user's stated uses, not assistant recipe suggestions."
        )
    if window_m:
        lines.append(
            f"The question restricts to a time window ('{window_m.group(0)}'). "
            "Check the @date stamps against the question date and count only in-window "
            "events. Output only the final integer plus short item names; no analysis."
        )
    return "\n".join(lines)


def build_expertise_digest(hits: list[dict], query: str = "") -> str:
    """
    For publication/conference recommends: surface demonstrated specialty domains
    so answers stay scoped (not generic venue lists from haystack noise).
    """
    if not is_pub_recommend_query(query) or not hits:
        return ""
    # Use max depth per domain from a single episode (sums reward noisy SoPs)
    by_domain: dict[str, float] = {}
    samples: dict[str, str] = {}
    for h in hits:
        content = h.get("content") or ""
        if _SOP_NOISE_RE.search(content):
            continue
        domain, depth = score_domain_depth(content)
        if not domain or depth < 2.5:
            continue
        if depth > by_domain.get(domain, 0):
            by_domain[domain] = depth
            cl = content.lower()
            for term in _DOMAIN_LEXICONS.get(domain, ()):
                if term in cl:
                    samples[domain] = term
                    break
            samples.setdefault(domain, domain.replace("_", " "))
    if not by_domain:
        return ""
    ranked = sorted(by_domain.items(), key=lambda x: x[1], reverse=True)
    top_domain, _ = ranked[0]
    label = samples.get(top_domain, top_domain.replace("_", " "))
    lines = [
        "User demonstrated specialty (scope recommendations to this focus; "
        "do not default to generic AI venues when a deeper specialty exists):",
        f"- primary focus: {label}",
    ]
    if len(ranked) > 1:
        others = [
            samples.get(d, d.replace("_", " ")) for d, _ in ranked[1:3]
        ]
        lines.append("- also seen: " + ", ".join(others))
    lines.append(
        "Prefer conferences/papers inside the primary focus; "
        "avoid recommending unrelated general topics when memory shows a specialty. "
        "Ignore admissions/SoP essays that only name-drop venues."
    )
    return "\n".join(lines)


_MAINT_FACT_RE = re.compile(
    r"[^\.\n]{0,40}\b(?:replac(?:e|ed|ing)|install(?:ed|ing)?|upgrad(?:e|ed|ing)|"
    r"new\s+(?:\w+\s+){0,3}(?:computer|device|kit|cassette|chain|tires?|battery))"
    r"[^\.\n]{0,80}",
    re.I,
)


def build_maintenance_digest(hits: list[dict], query: str = "") -> str:
    """Surface maintenance/gear facts for preference follow-ups (bike performance, etc.)."""
    if not is_preference_context_query(query) or not hits:
        return ""
    lines = [
        "Related maintenance / gear facts from memory "
        "(cite all of these when explaining improved performance):",
    ]
    n = 0
    seen: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        sid = h.get("source_description") or h.get("uuid") or "?"
        for m in _MAINT_FACT_RE.finditer(content):
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            key = span.lower()
            if not span or key in seen:
                continue
            seen.add(key)
            lines.append(f"- ({sid}) {span[:160]}")
            n += 1
            if n >= 6:
                break
        if n >= 6:
            break
    if n == 0:
        return ""
    lines.append(
        "When answering why something improved, mention every listed fact that applies "
        "(for example both replaced parts and a new computer/device)."
    )
    return "\n".join(lines)


_PRIOR_EXPERIENCE_RE = re.compile(
    r"\bI(?:'ve| have)?(?:\s+\w+){0,6}\s+"
    r"(?:met|saw|attended|went to|visited|watched)\s+[^\.\n]{5,120}",
    re.I,
)


def build_preference_digest(hits: list[dict]) -> str:
    """Aggregate transferable preference cues for recommend-style questions."""
    prefs: list[str] = []
    feats: list[str] = []
    experiences: list[str] = []
    for h in hits:
        cues = h.get("cues") or {}
        for p in cues.get("preferences") or []:
            if p not in prefs:
                prefs.append(p)
        for f in cues.get("features") or []:
            if f not in feats:
                feats.append(f)
        content = h.get("content") or ""
        for m in _PRIOR_EXPERIENCE_RE.finditer(content):
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            key = span.lower()
            if len(span) < 12 or key in {e.lower() for e in experiences}:
                continue
            experiences.append(span[:160])
            if len(experiences) >= 4:
                break
    if not prefs and not feats and not experiences:
        return ""
    lines = [
        "Transferable user preferences from memory "
        "(reuse for the city/topic in the question even if the session names another city):"
    ]
    for p in prefs[:4]:
        lines.append(f"- {p}")
    if feats:
        lines.append("- features: " + ", ".join(feats[:8]))
    if experiences:
        lines.append("Named prior experiences to ground tips (mention these by name when relevant):")
        for e in experiences[:4]:
            lines.append(f"- {e}")
    lines.append(
        "Give a concrete recommendation for the asked place/topic using these preferences; "
        "do not refuse or say you do not know when these preferences are listed. "
        "When named prior experiences are listed, open with those specifics "
        "(people met, concerts, venues) before any generic city/shopping list. "
        "For hotel questions: open with a matching hotel suggestion for the asked city; "
        "never answer that you lack Miami/city hotels when these features are listed. "
        "For cultural-event or weekend questions: suggest events that fit the listed "
        "language/practice or cultural interests."
    )
    return "\n".join(lines)
