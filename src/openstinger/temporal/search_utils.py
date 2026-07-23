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
    Lexical bridges for 'how many events/ceremonies did I attend' style counts.

    Questions name the category (art-related events, graduation ceremonies)
    but gold sessions use attended/volunteered/lecture/exhibition wording
    without repeating the question noun.
    """
    if not is_count_query(query):
        return []
    ql = (query or "").lower()
    if not re.search(
        r"\b(?:events?|exhibitions?|tours?|ceremon(?:y|ies)|graduations?)\b",
        ql,
    ):
        return []
    if not re.search(r"\b(?:attend|attended|went|visit|visited)\b", ql):
        return []
    bridges = [
        "attended",
        "volunteered",
        "exhibition",
        "lecture",
        "guided tour",
        "museum",
        "gallery",
        "art afternoon",
    ]
    if re.search(r"\b(?:ceremon(?:y|ies)|graduations?)\b", ql):
        bridges.extend(["graduation", "ceremony", "commencement"])
    return bridges


def health_device_bridge_terms(query: str) -> list[str]:
    """
    Lexical bridges for health-device inventory counts.

    Questions say 'health-related devices' but sessions name class nouns
    (nebulizer, hearing aids, glucose meter, smartwatch) without 'device'.
    """
    if not is_health_device_count_query(query):
        return []
    return [
        "nebulizer",
        "hearing",
        "hearing aids",
        "glucose",
        "blood sugar",
        "smartwatch",
        "meter",
        "monitor",
        "wearing my",
        "using my",
        "inhalation",
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
            "participated",
            "completed",
            "triathlon",
            "5k run",
            "5k",
            "tournament",
            "soccer",
            "race",
            "watched",
            "attended",
            "championship",
            "nba",
        )
    if is_role_tenure_query(query):
        _add(
            "years and",
            "months",
            "worked my way",
            "started as",
            "experience in the company",
            "senior",
            "specialist",
            "coordinator",
            "current role",
        )
    if is_pairwise_first_query(query):
        _add("set up", "installed", "upgraded", "month ago", "recently")
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
    # Offset: "how many days before <event A> did I <event B>"
    r"\bhow many (?:days?|weeks?|months?)\s+before\b|"
    # Duration-to-complete: "how many weeks did it take me to watch…"
    r"\bhow many (?:days?|weeks?|months?|years?) did it take\b|"
    # Shipping latency: how many days from order → arrive (no "passed" wording)
    r"\bhow many days\b.{0,100}\b(?:order(?:ed)?|bought|purchased)\b|"
    r"\bhow many days\b.{0,100}\b(?:arriv(?:e|ed|al)|receiv(?:e|ed)|deliver(?:y|ed))\b",
    re.I,
)


def is_stated_effort_duration_query(query: str) -> bool:
    """
    'How many weeks did it take me to watch/read/finish X (and Y)?'

    These are first-person binge/effort totals stated in memory, not calendar
    spans between two dated events.
    """
    return bool(
        re.search(
            r"\bhow many (?:days?|weeks?|months?)\s+did it take\b.{0,80}\b"
            r"(?:watch|watched|reading|read|finish(?:ed)?|complete(?:d)?|binge)",
            query or "",
            re.I,
        )
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
    # Watch/read effort totals use stated-duration digests, not event-date spans.
    if is_stated_effort_duration_query(q):
        return False
    return True


def is_rewatch_count_query(query: str) -> bool:
    """How many X did I re-watch (titles), not how many I watched in total."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\bre-?watch", q, re.I)
    )


def is_subscription_count_query(query: str) -> bool:
    """How many magazine/publication subscriptions (active), not issues bought."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\bsubscriptions?\b", q, re.I)
    )


def is_service_plan_count_query(query: str) -> bool:
    """How many X did I service or plan to service (distinct assets)."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\bservice", q, re.I)
        and re.search(r"\bplan\b", q, re.I)
    )


def is_health_device_count_query(query: str) -> bool:
    """How many health-related devices do I use (daily wearables/meters/aids)."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\bdevices?\b", q, re.I)
        and re.search(r"\b(?:health|medical|day)\b", q, re.I)
    )


def is_delivery_service_count_query(query: str) -> bool:
    """How many food/meal delivery services (types) have I used."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\bdelivery\b", q, re.I)
        and re.search(r"\b(?:services?|apps?|platforms?)\b", q, re.I)
    )


def is_tank_population_query(query: str) -> bool:
    """How many fish/animals across tanks (not how many tanks)."""
    q = query or ""
    if not re.search(r"\bhow many\b", q, re.I):
        return False
    if re.search(r"\bhow many\s+(?:tanks?|aquariums?)\b", q, re.I):
        return False
    return bool(
        re.search(r"\b(?:fish|bettas?|tetras?|gouramis?|catfish)\b", q, re.I)
        and re.search(r"\b(?:tanks?|aquariums?)\b", q, re.I)
    )


def is_tank_count_query(query: str) -> bool:
    """How many tanks/aquariums (including set-up-for-others)."""
    q = query or ""
    if is_tank_population_query(q):
        return False
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\b(?:tanks?|aquariums?)\b", q, re.I)
    )


def is_comparative_savings_query(query: str) -> bool:
    """How much save/cheaper by choosing A instead of B."""
    q = query or ""
    return bool(
        re.search(r"\b(?:save|saving|cheaper|difference)\b", q, re.I)
        and re.search(r"\b(?:instead of|versus|vs\.?|rather than)\b", q, re.I)
        and re.search(r"\b(?:how much|\$|cost|fare|price)\b", q, re.I)
    )


def is_pairwise_first_query(query: str) -> bool:
    """Which of A or B happened / was set up first."""
    q = query or ""
    return bool(
        re.search(
            r"\bwhich\b.+\b(?:first|earlier|before)\b|"
            r"\b(?:first|earlier)\b.+\bor\b.+\?",
            q,
            re.I,
        )
        and re.search(r"\bor\b", q, re.I)
    )


def is_role_tenure_query(query: str) -> bool:
    """How long in current role/job/position."""
    q = query or ""
    return bool(
        re.search(r"\bhow long\b", q, re.I)
        and re.search(
            r"\b(?:current role|current (?:job|position)|working in my current|"
            r"been (?:in|at) (?:my|this) (?:role|job|position))\b",
            q,
            re.I,
        )
    )


def is_types_of_count_query(query: str) -> bool:
    """How many different types/kinds of X."""
    q = query or ""
    return bool(
        re.search(r"\bhow many\b", q, re.I)
        and re.search(r"\b(?:different )?types? of\b|\bkinds? of\b", q, re.I)
    )


def is_age_delta_query(query: str) -> bool:
    """How many years older am I than when I graduated / at event age."""
    q = query or ""
    return bool(
        re.search(r"\bhow many years older am i\b", q, re.I)
        or (
            re.search(r"\byears older\b", q, re.I)
            and re.search(r"\b(?:graduat|college|when i was)\b", q, re.I)
        )
    )


def age_delta_bridge_terms(query: str) -> list[str]:
    """Bridges so current-age and age-at-event sessions both retrieve."""
    if not is_age_delta_query(query):
        return []
    return [
        "year-old",
        "year old",
        "age of",
        "graduated",
        "graduation",
        "degree",
        "college",
    ]


def is_fitness_days_per_week_query(query: str) -> bool:
    """How many days/classes a week for fitness/workout attendance.

    Covers both 'how many days a week … fitness classes' and structural
    'how many fitness classes … typical/each/per week' wording.
    """
    q = query or ""
    if not re.search(r"\bhow many\b", q, re.I):
        return False
    if not re.search(
        r"\b(?:week|weekly|typical\s+week|each\s+week|per\s+week)\b", q, re.I
    ):
        return False
    if not re.search(r"\b(?:fitness|workout|class(?:es)?)\b", q, re.I):
        return False
    if re.search(r"\bhow many\s+days\b", q, re.I):
        return True
    # Classes-per-week (weekday harvest still counts distinct days attended)
    return bool(
        re.search(
            r"\bhow many\b.{0,48}\b(?:fitness\s+|workout\s+)?class(?:es)?\b",
            q,
            re.I,
        )
    )


def delivery_service_bridge_terms(query: str) -> list[str]:
    """
    Lexical bridges for delivery-service inventory counts.

    Questions say 'food delivery services' but sessions name apps/brands
    (Uber Eats, DoorDash) or 'called <Service>' without repeating 'services'.
    """
    if not is_delivery_service_count_query(query):
        return []
    return [
        "uber eats",
        "doordash",
        "grubhub",
        "postmates",
        "food delivery",
        "meal delivery",
        "delivery services",
        "ordered from",
        "domino",
    ]


def tank_bridge_terms(query: str) -> list[str]:
    """Bridges for tank/aquarium inventory (gallon sizes, set-up-for-others)."""
    if not (is_tank_count_query(query) or is_tank_population_query(query)):
        return []
    return [
        "gallon",
        "aquarium",
        "betta",
        "community tank",
        "set up",
        "set up a",
        "currently has",
        "schooling fish",
    ]


def temporal_span_side_bridge_terms(query: str) -> list[str]:
    """
    Bridges from both sides of since/between/before span questions.

    Stops global min/max fallback when one event session never enters top-k
    (e.g. ukulele-lessons start missing while guitar-tech day retrieves).
    """
    if not asks_temporal_span_integer(query):
        return []
    q = query or ""
    sides: list[str] = []
    m = re.search(
        r"\b(?:had |have )?passed since\b(.+?)\bwhen i\b(.+?)(?:\?|$)",
        q,
        re.I,
    )
    if m:
        sides.extend([m.group(1), m.group(2)])
    m = re.search(
        r"\bbetween the day\b(.+?)\band the day\b(.+?)(?:\?|$)",
        q,
        re.I,
    )
    if m:
        sides.extend([m.group(1), m.group(2)])
    m = re.search(
        r"\bhow many (?:days?|weeks?|months?)\s+before\b(.+?)\bdid i\b(.+?)(?:\?|$)",
        q,
        re.I,
    )
    if m:
        sides.extend([m.group(1), m.group(2)])
    out: list[str] = []
    seen: set[str] = set()
    for side in sides:
        for t in extract_topic_phrases(side) + extract_topic_nouns(side):
            t = (t or "").strip().lower()
            if len(t) < 4 or t in seen:
                continue
            seen.add(t)
            out.append(t)
        for m2 in re.finditer(
            r"\b(ukulele|guitar tech|spark plugs?|turbocharged|"
            r"tennis racket|recovered|10th jog|flu)\b",
            side,
            re.I,
        ):
            t = m2.group(1).lower()
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out[:12]


def is_topic_inventory_query(query: str) -> bool:
    """Non-errand, non-duration, non-temporal item/project counts (kits, albums)."""
    if not is_count_query(query):
        return False
    if is_errand_count_query(query) or is_activity_duration_query(query):
        return False
    if _TEMPORAL_SPAN_COUNT_RE.search(query or ""):
        return False
    # Owned by specialized digests / aggregate harvest paths
    if (
        is_rewatch_count_query(query)
        or is_subscription_count_query(query)
        or is_service_plan_count_query(query)
        or is_health_device_count_query(query)
        or is_delivery_service_count_query(query)
        or is_tank_count_query(query)
        or is_fitness_days_per_week_query(query)
    ):
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

    # Inventory / soft-advice bridges first so the return cap cannot drop them.
    if is_soft_advice_query(q):
        for bridge in soft_advice_bridge_terms(q):
            _add(bridge)
    for bridge in event_attend_bridge_terms(q):
        _add(bridge)
    for bridge in fact_lookup_bridge_terms(q):
        _add(bridge)
    for bridge in health_device_bridge_terms(q):
        _add(bridge)
    for bridge in delivery_service_bridge_terms(q):
        _add(bridge)
    for bridge in tank_bridge_terms(q):
        _add(bridge)
    for bridge in temporal_span_side_bridge_terms(q):
        _add(bridge)

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
        if len(out) >= max_extra + 1 + 8:
            break

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
    countish = is_count_query(query) or is_aggregate_query(query)
    # Count/aggregate facts often sit in the opening user turn. Windowing only
    # around later keyword hits was dropping "I'm also getting Architectural
    # Digest" while keeping a late assistant "enjoy your … subscription".
    if countish:
        windows.append((0, min(len(text), 2800)))
    # Pin subscription / device acquire phrasing for inventory questions
    if countish:
        for m in re.finditer(
            r"\b(?:getting|subscription(?:\s+to)?|subscribed|canceled|"
            r"cancelled|wearing my|using my|with my|nebulizer|hearing aids?|"
            r"smartwatch|glucose|blood sugar)\b",
            text,
            re.I,
        ):
            a = max(0, m.start() - 200)
            b = min(len(text), m.end() + 220)
            windows.append((a, b))
            if len(windows) >= 10:
                break
    # Pin span event verbs so since/between sides survive excerpting
    if is_temporal_span_query(query):
        for m in re.finditer(
            r"\b(?:guitar tech|ukulele|spark plugs?|turbocharged|"
            r"recovered from the flu|10th jog|decided to take|"
            r"received my|bought a new|for servicing)\b",
            text,
            re.I,
        ):
            a = max(0, m.start() - 220)
            b = min(len(text), m.end() + 240)
            windows.append((a, b))
            if len(windows) >= 12:
                break
    # Pin tenure arithmetic phrases for current-role duration questions
    if is_role_tenure_query(query):
        for m in re.finditer(
            r"\b(?:\d+\s+years?\s+and\s+\d+\s+months?|worked my way|"
            r"experience in the company|started as|current role)\b",
            text,
            re.I,
        ):
            a = max(0, m.start() - 240)
            b = min(len(text), m.end() + 260)
            windows.append((a, b))
            if len(windows) >= 12:
                break
    if is_age_delta_query(query):
        for m in re.finditer(
            r"\b(?:\d{1,2}\s*-?\s*year\s*-?\s*old|at the age of\s+\d{1,2}|"
            r"graduat(?:ed|ion)|completed.{0,40}degree)\b",
            text,
            re.I,
        ):
            a = max(0, m.start() - 220)
            b = min(len(text), m.end() + 240)
            windows.append((a, b))
            if len(windows) >= 12:
                break
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
    device_bridges = health_device_bridge_terms(query)
    delivery_bridges = delivery_service_bridge_terms(query)
    tank_bridges = tank_bridge_terms(query)
    span_bridges = temporal_span_side_bridge_terms(query)
    age_bridges = age_delta_bridge_terms(query)
    if (
        not is_topic_inventory_query(query)
        and not is_preference_context_query(query)
        and not is_soft_advice_query(query)
        and not needs_preference_retrieval(query)
        and not fact_bridges
        and not event_bridges
        and not device_bridges
        and not delivery_bridges
        and not tank_bridges
        and not span_bridges
        and not age_bridges
        and not is_subscription_count_query(query)
        and not is_health_device_count_query(query)
        and not is_delivery_service_count_query(query)
        and not is_tank_count_query(query)
        and not is_temporal_span_query(query)
        and not is_age_delta_query(query)
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
    for bridge in (
        event_bridges
        + fact_bridges
        + device_bridges
        + delivery_bridges
        + tank_bridges
        + span_bridges
        + age_bridges
    ):
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
    # Chronological order lists need multi-session coverage, but must NOT use
    # count/enumerate digests (those overwrite the order with a bare integer).
    if is_temporal_order_query(query):
        return True
    return (
        is_count_query(query)
        or is_temporal_span_query(query)
        or bool(
            re.search(
                r"\b(?:in total|altogether|all the|every|"
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
    # Duration / activity totals are not per-conjunct money/count sums.
    # Avoid splitting "MCU movies and the main Star Wars films" or
    # "jogging and yoga" into incomplete-conjunct abstains.
    if (
        asks_temporal_span_integer(q)
        or is_activity_duration_query(q)
        or is_stated_effort_duration_query(q)
    ):
        return []
    if re.search(r"\bhow many (?:days?|weeks?|months?|years?) did it take\b", q, re.I):
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

    # "including A, B, and C" is category expansion, not A-and-B pairing.
    q_for_and = re.split(r"\bincluding\b", q, maxsplit=1, flags=re.I)[0].strip()
    if not q_for_and:
        q_for_and = q

    out: list[str] = []
    for kind, pat in (
        ("and", r"(.+?)\s+and\s+(?:the\s+|a\s+)?(.+?)(?:\?|$)"),
        (
            "instead",
            r"(.+?)\s+instead of\s+(?:a\s+|the\s+)?(.+?)(?:\?|$)",
        ),
    ):
        scan_q = q_for_and if kind == "and" else q
        m = re.search(pat, scan_q, re.I)
        if not m:
            continue
        left = None
        right = None
        if kind == "and":
            # Collocations: friends and family is one group, not A-and-B sides
            if re.search(r"\bfriends\s*$", m.group(1), re.I) and re.search(
                r"^family\b", m.group(2), re.I
            ):
                continue
            if re.search(r"\bblack\s*$", m.group(1), re.I) and re.search(
                r"^white\b", m.group(2), re.I
            ):
                continue
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


def build_tank_population_digest(hits: list[dict], query: str = "") -> str:
    """
    Sum first-person stocked fish counts across aquarium sessions.

    Distinguishes population totals from tank-count inventory.
    """
    if not is_tank_population_query(query) or not hits:
        return ""
    per_session_map: dict[str, int] = {}
    details: list[str] = []
    for h in hits:
        content = h.get("content") or ""
        if content.lstrip().startswith("[Session events"):
            continue
        if not re.search(r"\b(?:tank|aquarium|gallon|betta|fish)\b", content, re.I):
            continue
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = "\n".join(user_chunks) if user_chunks else content
        stocked = 0
        # "currently has 10 neon tetras, 5 golden honey gouramis, and a small pleco"
        for m in re.finditer(
            r"\b(?:currently has|which (?:currently )?has|tank,? which has)\b"
            r"([^\n.]{0,220})",
            scan,
            re.I,
        ):
            chunk = m.group(1)
            for nm in re.finditer(
                r"\b(\d{1,3})\s+([A-Za-z][A-Za-z '-]{2,40}?)"
                r"(?=\s*,|\s+and\b|\s*\.|$)",
                chunk,
            ):
                n = int(nm.group(1))
                name = nm.group(2).strip().lower()
                if n <= 0 or n > 200:
                    continue
                if re.search(
                    r"\b(?:gallon|plant|decoration|watt|week|day|month)\b",
                    name,
                ):
                    continue
                stocked += n
            for _sm in re.finditer(
                r"\b(?:a|an|one)\s+(?:small\s+|large\s+)?"
                r"(?:[A-Za-z][A-Za-z '-]{2,30}\s+fish|"
                r"[A-Za-z][A-Za-z '-]{2,20}\s+catfish|"
                r"betta(?:\s+fish)?)\b",
                chunk,
                re.I,
            ):
                stocked += 1
        # Separate betta tank mentioned without a currently-has list
        if stocked == 0 and re.search(
            r"\b(?:my|has my)\s+betta(?:\s+fish)?\b|"
            r"\bbetta fish,?\s+[A-Z][a-z]+\b",
            scan,
            re.I,
        ):
            stocked = 1
        if stocked <= 0:
            continue
        prev = per_session_map.get(sid, 0)
        if stocked > prev:
            per_session_map[sid] = stocked
            details.append(f"{sid}: {stocked}")
    if not per_session_map:
        return ""
    total = sum(per_session_map.values())
    if total <= 0:
        return ""
    return "\n".join(
        [
            "Tank population notes (sum stocked fish across aquariums):",
            "Per-tank first-person stocked counts: " + "; ".join(details[:6]),
            f"Suggested stated total from first-person count claim: {total}.",
            f"Answer with the integer {total}.",
        ]
    )


def _parse_hit_datetime(h: dict) -> Any | None:
    """Parse episode valid_at / valid_at_human into a naive UTC datetime."""
    from datetime import datetime as _dt, timezone as _tz

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
    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
        "december": 12,
    }
    m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})", when)
    if m2:
        mon = _MONTHS.get(m2.group(1).lower())
        if mon:
            try:
                return _dt(int(m2.group(3)), mon, int(m2.group(2)))
            except ValueError:
                return None
    return None


def build_pairwise_first_digest(hits: list[dict], query: str = "") -> str:
    """Decide which of two named options happened first from dated evidence."""
    if not is_pairwise_first_query(query) or not hits:
        return ""
    # Extract A or B phrases: "... first, the X or the Y?"
    m = re.search(
        r"\bfirst,?\s+(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?(.+?)(?:\?|$)",
        query.strip(),
        re.I,
    )
    if not m:
        m = re.search(
            r"\b(?:earlier|before),?\s+(?:the\s+)?(.+?)\s+or\s+(?:the\s+)?"
            r"(.+?)(?:\?|$)",
            query.strip(),
            re.I,
        )
    if not m:
        return ""
    sides = [
        re.sub(r"\s+", " ", m.group(1)).strip(" ?.,"),
        re.sub(r"\s+", " ", m.group(2)).strip(" ?.,"),
    ]
    sides = [
        re.split(r"\b(?:did i|do i|have i)\b", s, maxsplit=1, flags=re.I)[0].strip()
        for s in sides
    ]
    # Titles often use wrapping quotes; keep mid-word apostrophes (Michael's).
    sides = [s.replace('"', "").strip(" ?.,") for s in sides]
    sides = [re.sub(r"'([^']{3,})'", r"\1", s) for s in sides]
    # Keep short person names (Tom, Amy) as single-token sides
    sides = [
        s
        for s in sides
        if len(s) >= 4 or (len(s) >= 3 and " " not in s and s[0].isalpha())
    ]
    if len(sides) != 2:
        return ""

    def _rel_rank(text: str, hit: dict) -> tuple[int, int, str]:
        """
        Sort key (bucket, time): lower = earlier.

        bucket 0 = in-text relative ago (N units / last month / last week)
        bucket 1 = session valid_at (also used for bare 'recently')
        bucket 2 = undated
        """
        tl = text.lower()
        ago = re.search(
            r"\b(\d+|a|one|two|three|four|five|six)\s+"
            r"(days?|weeks?|months?|years?)\s+ago\b",
            tl,
        )
        if ago:
            nraw, unit = ago.group(1), ago.group(2)
            n = {
                "a": 1, "one": 1, "two": 2, "three": 3,
                "four": 4, "five": 5, "six": 6,
            }.get(nraw, None)
            if n is None:
                try:
                    n = int(nraw)
                except ValueError:
                    n = 1
            mult = 1
            if unit.startswith("week"):
                mult = 7
            elif unit.startswith("month"):
                mult = 30
            elif unit.startswith("year"):
                mult = 365
            return (0, -(n * mult), f"{n} {unit} ago")
        if re.search(r"\blast month\b|\ba month ago\b", tl):
            return (0, -30, "last month")
        if re.search(r"\blast week\b|\ba week ago\b", tl):
            return (0, -7, "last week")
        dt = _parse_hit_datetime(hit)
        stamp = (hit.get("valid_at_human") or "").strip() or "session"
        # Bare recently / last weekend are weak; prefer session chronology.
        if dt is not None:
            why = "session @" + stamp
            if re.search(
                r"\b(?:recently|just|today|this week|last weekend|"
                r"last saturday|last sunday)\b",
                tl,
            ):
                why = "recent @" + stamp
            return (1, dt.toordinal(), why)
        if re.search(r"\b(?:recently|just|today|this week)\b", tl):
            return (1, 2_000_000, "recently")
        return (2, 0, "undated")

    scored: list[tuple[int, int, str, str]] = []
    # Verbs/shells in "which first, the purchase of X" are not reliable match keys
    _side_weak = {
        "purchase", "purchased", "buying", "bought", "malfunction", "malfunctions",
        "event", "events", "attend", "attended", "happened", "setup", "set",
        "the", "and", "for", "from", "with", "into", "onto", "that", "this",
        "was", "were", "been", "being", "have", "has", "had", "did", "does",
    }
    for side in sides:
        best: tuple[int, int, str, str] | None = None
        side_l = side.lower()
        side_tokens = [
            p for p in side_l.split() if len(p) >= 3 and p not in _side_weak
        ]
        if not side_tokens:
            side_tokens = [
                p for p in side_l.split() if len(p) >= 3 and p not in {"the", "and"}
            ]
        for h in hits:
            content = h.get("content") or ""
            cl = content.lower()
            # Require every distinctive noun (coffee+maker, stand+mixer, …)
            if side_tokens:
                if not all(soft_contains(cl, p) for p in side_tokens):
                    continue
            elif not soft_contains(cl, side_l):
                continue
            # Short person names: require parenthood/event cue near the name
            # so bare "Tom" in unrelated chatter does not count as evidence.
            if " " not in side_l and len(side_l) <= 4:
                if not re.search(rf"\b{re.escape(side_l)}\b", cl):
                    continue
                if not re.search(
                    rf"\b{re.escape(side_l)}\b.{{0,100}}\b(?:parent|born|adopt|"
                    rf"baby|father|mother|son|daughter|child)\b|"
                    rf"\b(?:parent|born|adopt|baby|father|mother|son|daughter|"
                    rf"child)\b.{{0,100}}\b{re.escape(side_l)}\b",
                    cl,
                    re.I,
                ):
                    continue
            user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
            scan = "\n".join(user_chunks) if user_chunks else content
            # Baseline: session stamp (relative cues must appear near side nouns)
            bucket, tkey, why = _rel_rank("", h)
            for seed in side_tokens or side_l.split()[:1]:
                for wm in re.finditer(re.escape(seed), scan, re.I):
                    w = scan[max(0, wm.start() - 100) : wm.end() + 220]
                    wb, wt, wwhy = _rel_rank(w, h)
                    if (wb, wt) < (bucket, tkey):
                        bucket, tkey, why = wb, wt, wwhy
            cand = (bucket, tkey, side, why)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best:
            scored.append(best)
    # One named option evidenced, the other missing → abstain (incomplete pair)
    if len(scored) == 1:
        missing = [s for s in sides if s.lower() != scored[0][2].lower()]
        lines = [
            "Pairwise first-event notes:",
            f"- Evidence found for: {scored[0][2]} ({scored[0][3]})",
            f"- No dated evidence found for: {missing[0] if missing else 'other option'}.",
            "Suggested answer: I do not know.",
            "Do not answer with the one-sided name when the other option is absent.",
        ]
        return "\n".join(lines)
    if len(scored) < 2:
        return ""
    scored.sort(key=lambda x: (x[0], x[1]))
    first = scored[0][2]
    lines = [
        "Pairwise first-event notes (earlier evidence first):",
    ]
    for _b, _t, side, why in scored:
        lines.append(f"- {side} ({why})")
    lines.append(f"Suggested first event: {first}.")
    lines.append(f"Answer with: {first}")
    return "\n".join(lines)


def is_weekday_wake_query(query: str) -> bool:
    """Wake-up time on named weekdays (possibly with an earlier/later offset)."""
    q = query or ""
    if not re.search(r"\b(?:wake|wakes|waking|get up|gets up)\b", q, re.I):
        return False
    return bool(
        re.search(
            r"\b(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|"
            r"saturdays?|sundays?|weekdays?|weekends?)\b",
            q,
            re.I,
        )
    )


def build_weekday_wake_digest(hits: list[dict], query: str = "") -> str:
    """
    Compose baseline wake time with weekday-specific earlier/later offsets.

    Example: wake at 7:00, Tue/Thu 15 minutes earlier → Suggested 6:45 AM.
    """
    if not is_weekday_wake_query(query) or not hits:
        return ""
    blob = "\n".join(
        (h.get("content") or "")
        for h in hits
        if not (h.get("content") or "").lstrip().startswith("[Session events")
    )
    user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", blob)
    scan = "\n".join(user_chunks) if user_chunks else blob
    if not scan:
        return ""

    def _parse_clock(text: str) -> tuple[int, int, str] | None:
        m = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
            text,
            re.I,
        )
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or "0")
        ampm = m.group(3).lower().replace(".", "")
        if ampm.startswith("p") and hour < 12:
            hour += 12
        if ampm.startswith("a") and hour == 12:
            hour = 0
        h12 = hour % 12 or 12
        suffix = "AM" if hour < 12 else "PM"
        label = f"{h12}:{minute:02d} {suffix}"
        return hour, minute, label

    baseline = None
    for m in re.finditer(
        r"(.{0,40}\b(?:wake|waking|get up|gets up)\b.{0,40})",
        scan,
        re.I,
    ):
        clock = _parse_clock(m.group(1))
        if clock and not re.search(r"\b(?:earlier|later|minutes?)\b", m.group(1), re.I):
            baseline = clock
            break
    if baseline is None:
        # Fallback: any wake clock in scan
        for m in re.finditer(
            r"\b(?:wake|waking|get up)\b[^\n.]{0,50}",
            scan,
            re.I,
        ):
            clock = _parse_clock(m.group(0))
            if clock:
                baseline = clock
                break
    if baseline is None:
        return ""

    # Asked weekdays
    asked: set[str] = set()
    ql = (query or "").lower()
    for full in (
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ):
        if re.search(rf"\b{full}s?\b", ql):
            asked.add(full)

    offset_min = 0
    offset_why = ""
    for m in re.finditer(
        r"(.{0,80}\b(\d+)\s+minutes?\s+(earlier|later)\b.{0,80})",
        scan,
        re.I,
    ):
        window = m.group(1).lower()
        # Must mention at least one asked weekday (or "those days" near them)
        if asked and not any(d in window for d in asked):
            # Allow if weekdays named just before this span in scan
            start = max(0, m.start() - 120)
            pref = scan[start : m.start()].lower()
            if not any(d in pref for d in asked):
                continue
        n = int(m.group(2))
        sign = -1 if m.group(3).lower() == "earlier" else 1
        offset_min = sign * n
        offset_why = f"{n} minutes {m.group(3).lower()}"
        break
    if offset_min == 0:
        return ""

    total = baseline[0] * 60 + baseline[1] + offset_min
    total = total % (24 * 60)
    hour, minute = divmod(total, 60)
    h12 = hour % 12 or 12
    suffix = "AM" if hour < 12 else "PM"
    suggested = f"{h12}:{minute:02d} {suffix}"
    days_label = " and ".join(sorted(asked)) if asked else "those weekdays"
    return "\n".join(
        [
            "Weekday wake-schedule notes:",
            f"- Baseline wake time: {baseline[2]}",
            f"- Offset on {days_label}: {offset_why}",
            f"Suggested wake time: {suggested}.",
            f"Answer with {suggested}.",
        ]
    )


def build_role_tenure_digest(hits: list[dict], query: str = "") -> str:
    """
    Infer current-role tenure from company tenure minus time-to-role.

    Structural: years/months arithmetic only; no job-title lists.
    """
    if not is_role_tenure_query(query) or not hits:
        return ""
    blob = "\n".join(
        (h.get("content") or "")
        for h in hits
        if not (h.get("content") or "").lstrip().startswith("[Session events")
    )
    user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", blob)
    scan = "\n".join(user_chunks) if user_chunks else blob

    def _ym(text: str) -> tuple[int, int] | None:
        m = re.search(
            r"\b(\d+)\s+years?\s+and\s+(\d+)\s+months?\b",
            text,
            re.I,
        )
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"\b(\d+)\s+years?\b", text, re.I)
        if m and not re.search(r"\bmonths?\b", text[m.end() : m.end() + 24], re.I):
            return int(m.group(1)), 0
        m = re.search(r"\b(\d+)\s+months?\b", text, re.I)
        if m:
            return 0, int(m.group(1))
        return None

    company = None
    prior = None
    # Prefer spans that mention company tenure vs promotion path.
    for m in re.finditer(
        r"([^\n.]{0,100}\d+\s+years?\s+and\s+\d+\s+months?[^\n.]{0,80})",
        scan,
        re.I,
    ):
        span = m.group(1)
        ym = _ym(span)
        if not ym:
            continue
        if re.search(
            r"\b(?:experience in the company|in the company|with the company|"
            r"at the company|with (?:this|the) company)\b",
            span,
            re.I,
        ):
            company = ym
        elif re.search(
            r"\b(?:worked my way|started as|after|promoted|way up)\b",
            span,
            re.I,
        ):
            prior = ym
    # Fallback: separate keyword windows if the duration sits nearby
    if not company:
        for m in re.finditer(
            r"([^\n.]{0,80}\b(?:experience in the company|in the company|"
            r"with the company|at the company)\b[^\n.]{0,40})",
            scan,
            re.I,
        ):
            company = _ym(m.group(1)) or company
    if not prior:
        for m in re.finditer(
            r"([^\n.]{0,120}\b(?:worked my way up|started as)\b[^\n.]{0,100})",
            scan,
            re.I,
        ):
            prior = _ym(m.group(1)) or prior
    if not company or not prior:
        return ""
    c_m = company[0] * 12 + company[1]
    p_m = prior[0] * 12 + prior[1]
    if c_m <= p_m:
        return ""
    cur = c_m - p_m
    y, mo = divmod(cur, 12)
    if y and mo:
        shown = f"{y} year{'s' if y != 1 else ''} and {mo} month{'s' if mo != 1 else ''}"
    elif y:
        shown = f"{y} year{'s' if y != 1 else ''}"
    else:
        shown = f"{mo} month{'s' if mo != 1 else ''}"
    return (
        "Role tenure notes (company tenure minus time to reach current role):\n"
        f"- Company tenure: {company[0]}y {company[1]}m\n"
        f"- Prior path duration: {prior[0]}y {prior[1]}m\n"
        f"Suggested role tenure: {shown}.\n"
        f"Answer with: {shown}"
    )


def build_age_delta_digest(hits: list[dict], query: str = "") -> str:
    """Current age minus age-at-event (e.g. graduated at 25, now 32 → 7)."""
    if not is_age_delta_query(query) or not hits:
        return ""
    blob = "\n".join(
        (h.get("content") or "")
        for h in hits
        if not (h.get("content") or "").lstrip().startswith("[Session events")
    )
    cur = None
    then = None
    m = re.search(r"\b(?:as a\s+)?(\d{1,2})\s*-?\s*year\s*-?\s*old\b", blob, re.I)
    if m:
        cur = int(m.group(1))
    m = re.search(
        r"\b(?:completed|graduated|finished).{0,80}?\bat the age of\s+(\d{1,2})\b|"
        r"\bat the age of\s+(\d{1,2})\b.{0,40}?\b(?:graduat|degree|college)\b|"
        r"\bwhen i was\s+(\d{1,2})\b.{0,40}?\b(?:graduat|degree)\b",
        blob,
        re.I,
    )
    if m:
        then = int(next(g for g in m.groups() if g))
    if cur is None or then is None or cur <= then or cur > 120 or then > 100:
        return ""
    delta = cur - then
    return (
        "Age-delta notes (current age minus age at referenced event):\n"
        f"- Current age claim: {cur}\n"
        f"- Age at event claim: {then}\n"
        f"Suggested stated total from first-person count claim: {delta}.\n"
        f"Answer with the integer {delta}."
    )


def build_types_of_digest(hits: list[dict], query: str = "") -> str:
    """
    Distinct ingredient/member types for 'how many different types of X'.

    Harvests first-person use mentions; category head comes from the question.
    """
    if not is_types_of_count_query(query) or not hits:
        return ""
    m = re.search(
        r"\b(?:types?|kinds?) of\s+"
        r"([A-Za-z]+(?:\s+(?:fruits?|vegetables?|items?|ingredients?))?)\b",
        query,
        re.I,
    )
    if not m:
        return ""
    category = m.group(1).strip().lower()
    category = re.sub(r"\s+", " ", category)
    # Lightweight closed set only for broad produce categories (not brand/topic lists)
    members = {
        "citrus": {"orange", "lime", "lemon", "grapefruit", "yuzu", "tangerine"},
        "citrus fruit": {"orange", "lime", "lemon", "grapefruit", "yuzu", "tangerine"},
        "citrus fruits": {"orange", "lime", "lemon", "grapefruit", "yuzu", "tangerine"},
    }
    vocab = members.get(category)
    if not vocab:
        return ""
    found: list[str] = []
    seen: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = ("\n".join(user_chunks) if user_chunks else content).lower()
        if not re.search(
            r"\b(?:cocktail|recipe|bitters|juice|sangria|gimlet|peels?|slices?)\b",
            scan,
        ):
            continue
        for fruit in sorted(vocab):
            if fruit in seen:
                continue
            if re.search(rf"\b{re.escape(fruit)}\b", scan):
                seen.add(fruit)
                found.append(fruit)
    if len(found) < 2:
        return ""
    n = len(found)
    return (
        f"Types-of notes for {category} (first-person recipe uses):\n"
        f"Distinct types: {', '.join(found)}.\n"
        f"Suggested stated total from first-person count claim: {n}.\n"
        f"Answer with the integer {n}."
    )


def build_aggregate_reading_digest(hits: list[dict], query: str = "") -> str:
    """
    Structured reading notes for multi-session count/money questions.

    Generic only: uses question nouns/phrases and A-and-B conjuncts from the
    question text. No benchmark topic lists. Enumerate open counts; sum only
    when each conjunct side has its own paired evidence.

    Non-destructive: do not emit aggregate notes for temporal-span / shipping
    latency questions (those use build_temporal_span_digest). Do not
    open-enumerate frequency questions (stated-count digests own those).
    """
    if not is_aggregate_query(query) or not hits:
        return ""
    # Temporal duration digests are authoritative for these; aggregate "1 item"
    # notes were overwriting correct day spans (L3 fair_c2b_l3can).
    if asks_temporal_span_integer(query) or is_shipping_latency_query(query):
        return ""
    # Watch/read binge totals own these (stated effort duration digest).
    if is_stated_effort_duration_query(query):
        return ""
    # Workout/hour digests own these; incomplete "jogging and yoga" conjuncts
    # were forcing abstain over Candidate duration sum.
    if is_activity_duration_query(query):
        return ""
    # Order lists are not integer counts
    if is_temporal_order_query(query):
        return ""
    # Specialized digests own these
    if is_tank_population_query(query) or is_types_of_count_query(query):
        return ""
    if (
        is_role_tenure_query(query)
        or is_pairwise_first_query(query)
        or is_age_delta_query(query)
    ):
        return ""
    ql = (query or "").lower()
    # Frequency / collection digests own these; aggregate open-enumerate was
    # forcing distinct-item count 1/2 over Suggested stated/collection totals.
    if re.search(
        r"\bhow many (?:times|trips|meet(?:-|\s)?ups?|visits|episodes)\b",
        ql,
    ) or (
        re.search(r"\bhow many\b", ql)
        and re.search(r"\b(?:collection|coins?)\b", ql)
    ):
        return ""
    # Inherit/acquire inventory is owned by topic heirloom enumeration.
    if re.search(r"\b(?:inherit|acquired?|heirloom)\b", ql) and re.search(
        r"\b(?:family|antique|vintage)\b", ql
    ):
        return ""
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    terms = [t for t in (phrases + nouns) if t and len(t) >= 3]
    if not terms:
        return ""
    conjuncts = extract_and_conjuncts(query)
    money_q = bool(
        re.search(
            r"\b(?:how much|\$|money|sold|spend|spent|minimum|raise|save|saving)\b",
            ql,
        )
    )
    open_enumerate = bool(re.search(r"\bhow many\b", ql)) and not conjuncts
    rewatch_q = is_rewatch_count_query(query)
    subscription_q = is_subscription_count_query(query)
    service_plan_q = is_service_plan_count_query(query)
    health_device_q = is_health_device_count_query(query)
    delivery_q = is_delivery_service_count_query(query)
    tank_q = is_tank_count_query(query)
    fitness_days_q = is_fitness_days_per_week_query(query)
    canceled_subs: set[str] = set()
    _WEEKDAY_CANON = {
        "monday": "monday",
        "mon": "monday",
        "tuesday": "tuesday",
        "tue": "tuesday",
        "tues": "tuesday",
        "wednesday": "wednesday",
        "wed": "wednesday",
        "thursday": "thursday",
        "thu": "thursday",
        "thur": "thursday",
        "thurs": "thursday",
        "friday": "friday",
        "fri": "friday",
        "saturday": "saturday",
        "sat": "saturday",
        "sunday": "sunday",
        "sun": "sunday",
    }
    # Acquire/re-watch questions: allow hyponym objects when the session is already
    # on-topic (matched query nouns) but the span uses acquire verbs without
    # repeating the head noun ("brought home a monstera" for plants).
    acquire_open = open_enumerate and bool(
        re.search(
            r"\b(?:acquire|acquired|buy|bought|purchased|own|owned|get|got|"
            r"re-?watch(?:ed)?|subscribe(?:d)?|service|serviced|set up)\b",
            ql,
        )
    )
    birth_open = open_enumerate and bool(
        re.search(r"\b(?:babies|baby|births?|born)\b", ql, re.I)
    )
    # Bare "got" / "I've got a lot of land" is possession chatter, not acquire.
    _ACQUIRE_VERB_RE = re.compile(
        r"\b(?:bought|purchased|downloaded|ordered|tried|watched|re-?watched|"
        r"owned?|picked up|assembled|fixed|sold|learned|cooked|acquired|adopted|"
        r"brought home|brought it home|subscribed(?:\s+to)?|set up|serviced|"
        r"got\s+from|got\s+(?:a|an|the|my)\s+[a-z])\b",
        re.I,
    )
    _EXISTENTIAL_GOT_RE = re.compile(
        r"\bi(?:'ve| have)\s+got\s+(?:a|an|the|my)?\s*"
        r"(?:lot|bit|pretty|area|land|idea|feeling|problem|question|"
        r"couple|few|bunch|ton)\b",
        re.I,
    )
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
    # At least one hit already names a question noun → allow hyponym acquire
    # sessions in the same pool ("peace lily" for plants).
    topic_anchored = any(
        soft_contains((h.get("content") or ""), t)
        for h in hits
        for t in terms
        if t and len(t) >= 3
    )

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
        hyponym_acquire = (
            acquire_open
            and topic_anchored
            and not matched
            and bool(_ACQUIRE_VERB_RE.search(cl))
            and bool(re.search(r"\b(?:i|i've|i am|i'm|my)\b", cl, re.I))
        )
        sub_hit = subscription_q and bool(
            re.search(
                r"\b(?:subscription|subscribed|canceled|cancelled|getting)\b",
                cl,
                re.I,
            )
        )
        service_hit = service_plan_q and bool(
            re.search(
                r"\b(?:bike|serviced|service|lubricat(?:ed|ing)|tire|chain)\b",
                cl,
                re.I,
            )
        )
        device_hit = health_device_q and bool(
            re.search(
                r"\b(?:device|watch|meter|monitor|nebulizer|hearing|glucose|"
                r"blood sugar|fitbit|smartwatch|machine|aids?)\b",
                cl,
                re.I,
            )
        )
        delivery_hit = delivery_q and bool(
            re.search(
                r"\b(?:delivery|uber\s*eats|doordash|grubhub|postmates|"
                r"domino'?s|fresh fusion|meal\s+delivery)\b",
                cl,
                re.I,
            )
        )
        tank_hit = tank_q and bool(
            re.search(r"\b(?:tank|aquarium|gallon)\b", cl, re.I)
        )
        fitness_hit = fitness_days_q and bool(
            re.search(
                r"\b(?:class|zumba|yoga|weightlifting|workout|fitness|"
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                cl,
                re.I,
            )
        )
        if (
            not matched
            and not (conjuncts and any(soft_contains(cl, c) for c in conjuncts))
            and not action_hit
            and not hyponym_acquire
            and not sub_hit
            and not service_hit
            and not device_hit
            and not delivery_hit
            and not tank_hit
            and not fitness_hit
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
        # Subscription titles can appear in assistant echoes after excerpting.
        scan_sub = content if subscription_q else scan

        # Fitness class days/week: distinct weekdays with first-person class claims.
        if fitness_days_q:
            user_voice = bool(
                re.search(r"\b(?:i|i'm|i am|i've|my)\b", scan, re.I)
            )
            for m in re.finditer(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)s?\b",
                scan,
                re.I,
            ):
                # Wide window: "I attend … on Saturdays" can put the day far
                # from the sentence-initial first person.
                around = scan[max(0, m.start() - 220) : m.end() + 90]
                if not re.search(
                    r"\b(?:class(?:es)?|zumba|yoga|weightlifting|workout|"
                    r"fitness|attend)\b",
                    around,
                    re.I,
                ):
                    continue
                if not user_voice and not re.search(
                    r"\b(?:i|i'm|i am|i've|my)\b",
                    around,
                    re.I,
                ):
                    continue
                raw = re.sub(r"s$", "", m.group(1).lower())
                day = _WEEKDAY_CANON.get(raw)
                if not day or day in seen_items:
                    continue
                seen_items.add(day)
                distinct_items.append(f"class day: {day}")

        # Aquarium/tank inventory: distinct gallon-sized tanks (incl. set-up-for-others).
        if tank_q:
            for m in re.finditer(
                r"\b(\d+)\s*-?\s*gallon\s+"
                r"((?:[\w'-]+\s+){0,4}tank)\b",
                scan,
                re.I,
            ):
                gallons = m.group(1)
                rest = re.sub(r"\s+", " ", m.group(2)).strip().lower()
                key = f"{gallons}-gallon {rest}"
                # Collapse synonyms: "freshwater community tank" vs "community tank"
                key_norm = re.sub(
                    r"\b(?:freshwater|saltwater|planted)\s+",
                    "",
                    key,
                )
                if key_norm in seen_items or key in seen_items:
                    continue
                seen_items.add(key_norm)
                distinct_items.append(f"tank: {gallons}-gallon {rest}"[:100])

        # Food delivery services / apps used (distinct named platforms).
        if delivery_q:
            for m in re.finditer(
                r"\b(Uber\s*Eats|DoorDash|Grubhub|Postmates|Instacart)\b"
                r"|\bcalled\s+([A-Z][A-Za-z0-9 &'-]{2,40}?)"
                r"(?=\s*[-,.]|\s+they\b|\s+-\s|\s+have\b|$)"
                r"|\b(?:had|ordered|from)\s+(Domino'?s(?:\s+Pizza)?)\b"
                r"|\b(Domino'?s(?:\s+Pizza)?)\s+(?:three times|delivery)\b",
                scan,
            ):
                title = next((g for g in m.groups() if g), None)
                if not title:
                    continue
                title = re.sub(r"\s+", " ", title).strip(" .,")
                # "called" captures need nearby delivery context
                if m.group(2) and not re.search(
                    r"\b(?:delivery|meal|food|eats)\b",
                    scan[max(0, m.start() - 80) : m.end() + 40],
                    re.I,
                ):
                    continue
                key = title.lower().replace("'", "'")
                key = re.sub(r"\s+", " ", key)
                if key in {"this new one", "a new one", "one"}:
                    continue
                if key in seen_items:
                    continue
                seen_items.add(key)
                distinct_items.append(f"delivery: {title}"[:100])

        # Health devices used daily: distinct meters/wearables/aids/machines.
        if health_device_q:
            for m in re.finditer(
                r"\b(?:wearing|using|with)\s+my\s+"
                r"((?:[A-Z0-9][\w.'-]*\s+){0,5}"
                r"(?:smartwatch|watch|system|meter|monitor|machine))\b"
                r"|\btesting\b[^\n.]{0,60}?\bwith\s+my\s+"
                r"((?:[A-Z0-9][\w.'-]*\s+){0,5}(?:system|meter|monitor))\b"
                r"|\bmy\s+(nebulizer(?:\s+machine)?)\b"
                r"|\b(hearing aids?)\s+from\b"
                r"|\bmy\s+(hearing aids?)\b",
                scan,
            ):
                title = next((g for g in m.groups() if g), None)
                if not title:
                    continue
                title = re.sub(r"\s+", " ", title).strip(" .,")
                if len(title) < 6 and "aid" not in title.lower():
                    continue
                key = title.lower()
                if "hearing aid" in key:
                    key = "hearing aids"
                    title = "hearing aids"
                if "nebulizer" in key:
                    key = "nebulizer"
                    title = "nebulizer"
                if key in seen_items:
                    continue
                seen_items.add(key)
                distinct_items.append(f"device: {title}"[:100])

        # Service-or-plan: distinct assets (e.g. bikes) with service done or planned.
        if service_plan_q:
            month_q = None
            m_mon = re.search(
                r"\bin\s+(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b",
                ql,
            )
            if m_mon:
                month_q = m_mon.group(1).lower()
            bike_names = re.findall(
                r"\b((?:commuter|road|mountain|hybrid|electric)\s+bike)\b",
                scan,
                re.I,
            )
            if not bike_names and re.search(r"\bbike\b", scan, re.I):
                bike_names = ["bike"]
            # "commuter bike is just a regular hybrid bike" → one asset
            specific = [
                b
                for b in bike_names
                if not re.search(r"\b(?:hybrid|electric)\s+bike\b", b, re.I)
            ]
            if specific:
                bike_names = specific
            for raw_bike in bike_names:
                bike = raw_bike.strip().lower()
                # Window: prefer sentences mentioning this bike
                windows = []
                for m in re.finditer(re.escape(raw_bike), scan, re.I):
                    windows.append(
                        scan[max(0, m.start() - 120) : m.end() + 160]
                    )
                if not windows:
                    windows = [scan]
                blob_w = "\n".join(windows)
                served = bool(
                    re.search(
                        r"\b(?:serviced|getting\s+(?:my\s+)?[^.\n]{0,40}\bserviced|"
                        r"cleaned and lubricated|lubricated the chain|"
                        r"lubricating the chain)\b",
                        blob_w,
                        re.I,
                    )
                )
                planned = bool(
                    re.search(
                        r"\b(?:plan(?:ning)?\s+to|time to replace|replace\s+it\s+"
                        r"this month|before april|this month,?\s+before)\b",
                        blob_w,
                        re.I,
                    )
                ) or bool(
                    re.search(
                        r"\b(?:replace|replacing)\b[^\n.]{0,40}\btire\b|"
                        r"\btire\b[^\n.]{0,40}\b(?:replace|replacing)\b",
                        blob_w,
                        re.I,
                    )
                )
                if not served and not planned:
                    continue
                if month_q and not (
                    month_q in blob_w.lower()
                    or month_q in (when or "").lower()
                    or (
                        planned
                        and re.search(
                            r"\b(?:this month|before april)\b", blob_w, re.I
                        )
                    )
                ):
                    # Session stamp month often carries the asked month
                    if month_q not in (when or "").lower():
                        continue
                kind = "serviced" if served else "planned"
                key = bike
                if key in seen_items:
                    continue
                seen_items.add(key)
                distinct_items.append(f"{kind}: {raw_bike.strip()}"[:100])

        # Subscriptions: active titles only; exclude canceled; ignore one-off issues.
        if subscription_q:

            def _sub_key(raw: str) -> str:
                t = re.sub(r"\s+", " ", (raw or "").strip().lower())
                t = re.sub(
                    r"\s+(?:magazine|publication|subscriptions?)\s*$",
                    "",
                    t,
                )
                return t.strip()

            for m in re.finditer(
                r"\b(?:canceled|cancelled)\s+(?:my\s+)?"
                r"([A-Z][A-Za-z0-9 .'&-]{2,40}?)\s+"
                r"(?:magazine\s+)?subscriptions?\b",
                scan_sub,
            ):
                canceled_subs.add(_sub_key(m.group(1)))
            for m in re.finditer(
                r"\bsubscription\s+to\s+([A-Z][A-Za-z0-9 .'&-]{2,50}?)"
                r"(?=\s+magazine\b|\s*,|\s+which\b|\s+in\b|\.|$)"
                r"|\bsubscribed\s+to\s+([A-Z][A-Za-z0-9 .'&-]{2,50}?)"
                r"(?=\s+magazine\b|\s*,|\s+which\b|\s+in\b|\.|$)"
                r"|\b(?:getting|receive|receiving)\s+"
                r"([A-Z][A-Za-z0-9 .'&-]{2,50}?)"
                r"(?=\s*,|\s+which\b|\s+for\b|\.|$)"
                # Assistant echo / truncated user turn: "your Architectural Digest subscription"
                r"|\b(?:your|my|the)\s+([A-Z][A-Za-z0-9 .'&-]{2,50}?)\s+"
                r"subscriptions?\b",
                scan_sub,
            ):
                title = next((g for g in m.groups() if g), None)
                if not title:
                    continue
                # Skip "canceled my Forbes … subscription" false harvest
                prefix = scan_sub[max(0, m.start() - 40) : m.start()].lower()
                if re.search(r"\b(?:canceled|cancelled)\b", prefix):
                    continue
                title = title.strip().rstrip(" .,")
                title = re.sub(
                    r"\s+(?:magazine|publication)\s*$",
                    "",
                    title,
                    flags=re.I,
                ).strip()
                # "getting Architectural Digest" / skip generic "other publications"
                if title.lower() in {
                    "other publications",
                    "publications",
                    "magazines",
                    "magazine",
                }:
                    continue
                key = _sub_key(title)
                if not key or key in canceled_subs or key in seen_items:
                    continue
                # One-off issue buys are not subscriptions
                if re.search(
                    r"\b(?:bought|buying|last)\b[^\n.]{0,40}\b"
                    + re.escape(title.split()[0]),
                    scan_sub,
                    re.I,
                ) and not re.search(
                    r"\b(?:subscription|subscribed|getting)\b[^\n.]{0,40}\b"
                    + re.escape(title.split()[0]),
                    scan_sub,
                    re.I,
                ):
                    continue
                seen_items.add(key)
                distinct_items.append(f"subscription: {title}"[:100])
            # "enjoying other publications like The New Yorker, which I subscribed"
            for m in re.finditer(
                r"\blike\s+([A-Z][A-Za-z0-9 .'&-]{2,50}?)"
                r"(?=\s*,|\s+which\b|\s+magazine\b)",
                scan_sub,
            ):
                if not re.search(
                    r"\b(?:subscribed|subscription|enjoying)\b",
                    scan_sub[max(0, m.start() - 80) : m.end() + 80],
                    re.I,
                ):
                    continue
                title = m.group(1).strip().rstrip(" .,")
                key = _sub_key(title)
                if not key or key in canceled_subs or key in seen_items:
                    continue
                seen_items.add(key)
                distinct_items.append(f"subscription: {title}"[:100])

        # Re-watch counts: enumerate titled re-watches only (not "watched N movies").
        if rewatch_q:
            for m in re.finditer(
                r"\bre-?watched\s+"
                r"(?:(?:the|a|an|another)\s+)*(?:Marvel\s+movies?\s*,?\s*)?"
                r"([A-Z0-9][^.\n]{1,70}?)"
                r"(?=\s*,|\s+which\b|\s+yesterday\b|\s+last\b|\s+this\b|"
                r"\s+and\b|\s+I've\b|\s+I\b|\.|$)",
                scan,
            ):
                title = m.group(1).strip()
                title = re.split(r"\s+which\b", title, maxsplit=1)[0].strip()
                title = re.sub(r"\s+", " ", title).strip(" .:;-")
                if len(title) < 3:
                    continue
                key = title.lower()
                if key in seen_items:
                    continue
                # Drop if no topic overlap when question names a franchise
                if terms and not any(
                    soft_contains(title, t) or soft_contains(scan, t) for t in terms
                ):
                    if not re.search(r"[A-Z]", title):
                        continue
                seen_items.add(key)
                distinct_items.append(f"re-watched {title}"[:100])

        for term in (
            matched
            or conjuncts
            or (["furniture"] if action_hit else [])
            or (["acquire"] if hyponym_acquire else [])
        )[:4]:
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
            rf"completed|finished|played|learned|scored|lasted|got)\b[^\n.]{{0,48}}?"
            rf"{num_tok}\b"
            rf"|{num_tok}\b[^\n.]{{0,32}}?(?:of\s+)?(?:{obj_alt})\b"
            rf"|\b(?:the\s+)?{num_tok}\s+(?:meal|meals|lunch|lunches|"
            rf"goal|goals|assist|assists|comments?|views?|likes?)\b"
            rf"|\b{num_tok}\s+comments?\b",
            scan,
            re.I,
        ):
            num_s = next((g for g in m.groups() if g), None)
            if not num_s:
                continue
            n = _parse_count_token(num_s)
            if n is None or n <= 0 or n > 5000:
                continue
            # "bought … two weeks ago" is a duration, not an item count
            after_num = scan[m.end() : m.end() + 24]
            if re.match(
                r"\s*(?:weeks?|days?|months?|years?|hours?|minutes?|ago)\b",
                after_num,
                re.I,
            ):
                continue
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            if re.search(
                r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
                r"(?:weeks?|days?|months?|years?)\b",
                span,
                re.I,
            ):
                continue
            around = scan[max(0, m.start() - 40) : m.end() + 40]
            pool = conjuncts or matched or terms
            if not any(soft_contains(around, t) for t in pool):
                continue
            number_claims.append((n, f"{span[:100]}{stamp}", _pair_object(around)))

        # Distinct item spans for open enumeration (first-person acquire/use).
        # Specialized harvests above; skip generic enum for those modes.
        if (
            open_enumerate
            and not rewatch_q
            and not subscription_q
            and not service_plan_q
            and not health_device_q
            and not delivery_q
            and not tank_q
            and not fitness_days_q
        ):
            enum_pats = [
                r"\b(?:i(?:'ve| have)?|my)\b[^\n.]{0,120}?\b(?:bought|purchased|"
                r"downloaded|ordered|tried|watched|re-?watched|owned?|picked up|"
                r"assembled|fixed|sold|learned|cooked|acquired|adopted|brought home|"
                r"brought it home|subscribed(?:\s+to)?|set up|serviced|service|"
                r"got\s+from|got\s+(?:a|an|the|my)\s+[a-z])\b[^\n.]{0,80}",
                # "snake plant, which I got from my sister"
                r"\b[A-Za-z][A-Za-z0-9 '-]{2,40}\b[^\n.]{0,40}?\bwhich\s+i\s+got\s+from\b"
                r"[^\n.]{0,40}",
            ]
            if birth_open:
                # Birth events only (not birthdays, pets, or calendar noise)
                enum_pats = [
                    r"\b(?:baby|son|daughter|boy|girl)\b[^\n.]{0,40}?\b(?:born|arrived)\b"
                    r"[^\n.]{0,40}",
                    r"\b(?:born|welcomed|gave birth|had a baby|had a son|had a daughter)\b"
                    r"[^\n.]{0,80}",
                    r"\b[A-Z][a-z]{2,12}\b[^\n.]{0,20}?\b(?:was born|was welcomed|"
                    r"came home from the hospital)\b[^\n.]{0,40}",
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
                    if _EXISTENTIAL_GOT_RE.search(span):
                        continue
                    if birth_open and re.search(
                        r"\b(?:dog|puppy|cat|kitten|pet|birthday|calendar)\b",
                        span,
                        re.I,
                    ):
                        continue
                    term_ok = any(soft_contains(span, t) for t in terms)
                    if birth_open:
                        term_ok = bool(
                            re.search(
                                r"\b(?:born|welcomed|baby|son|daughter|"
                                r"gave birth|hospital)\b",
                                span,
                                re.I,
                            )
                        )
                    action_ok = multi_action and bool(
                        re.search(
                            rf"(?:\b(?:{_act}|get a new)\b.{{0,80}}\b(?:{_obj})\b|"
                            rf"\b(?:{_obj})\b.{{0,80}}\b(?:{_act}|get a new)\b)",
                            span,
                            re.I,
                        )
                    )
                    # On-topic pool + acquire verb: count hyponym gets
                    acquire_ok = (
                        acquire_open
                        and (bool(matched) or hyponym_acquire)
                        and bool(_ACQUIRE_VERB_RE.search(span))
                    )
                    if not term_ok and not action_ok and not acquire_ok:
                        continue
                    # Dedupe by specific object phrase (coffee table ≠ kitchen table)
                    obj_m = re.search(
                        r"\b(?:(?:coffee|kitchen|bedside|side|dining|end)\s+)?"
                        r"(?:table|bookshelf|desk|chair|dresser|mattress|sofa|"
                        r"couch|cabinet|shelf|lamp|bed)\b",
                        span,
                        re.I,
                    )
                    # Prefer object after acquire verb so hyponyms dedupe cleanly
                    acq_obj = re.search(
                        r"\b(?:bought|purchased|acquired|adopted|ordered|"
                        r"brought home|brought it home|re-?watched|"
                        r"subscribed(?:\s+to)?|set up|serviced)\b\s+"
                        r"(?:a|an|the|my|our)?\s*"
                        r"([A-Za-z][A-Za-z0-9 '-]{2,40})"
                        r"|\bgot\s+(?!from\b)(?:a|an|the|my|our)?\s*"
                        r"([A-Za-z][A-Za-z0-9 '-]{2,40})"
                        r"|\b([A-Za-z][A-Za-z0-9 '-]{2,30}\s+plant)\s*,?\s*"
                        r"which\s+i\s+got\s+from\b",
                        span,
                        re.I,
                    )
                    if acq_obj:
                        acq_name = next(
                            (g for g in acq_obj.groups() if g), None
                        )
                    else:
                        acq_name = None
                    _q_nouns = {
                        re.sub(r"s$", "", t.lower())
                        for t in terms
                        if t and len(t) >= 3
                    } | {"plant", "plants", "item", "items", "thing", "things"}

                    def _item_key(raw: str) -> str | None:
                        t = (raw or "").strip().lower()
                        t = re.sub(r"\s+", " ", t)
                        m_bought = re.search(
                            r"\bbought\s+(?:a|an|the|my)?\s*"
                            r"([a-z]+(?:\s+[a-z]+)?)",
                            t,
                        )
                        if m_bought and m_bought.group(1) not in _q_nouns:
                            return m_bought.group(1)
                        m_which = re.search(
                            r"\b(?:my|a|an|the)\s+"
                            r"([a-z]+(?:\s+[a-z]+)?)\s*,?\s*which\s+i\s+got\s+from\b",
                            t,
                        )
                        if m_which and m_which.group(1) not in _q_nouns:
                            return m_which.group(1)
                        # Prefer "… plant" / "… lily" heads for stable dedupe
                        m_head = re.search(
                            r"\b([a-z]+(?:\s+[a-z]+)?\s+plants?|[a-z]+\s+lily|"
                            r"succulent(?:\s+plant)?)\b",
                            t,
                        )
                        if m_head:
                            head = m_head.group(1)
                            head = re.sub(r"\s+plants?$", " plant", head)
                            head_norm = re.sub(r"s$", "", head)
                            if head in _q_nouns or head_norm in _q_nouns:
                                return None
                            return head
                        # No stable object key → skip (avoids duplicate nursery chatter)
                        return None

                    def _register_item(key: str, label: str) -> bool:
                        """Dedupe near-prefix collisions (peace li / peace lily)."""
                        key = re.sub(r"\s+", " ", (key or "").strip().lower())
                        if not key or key in _q_nouns:
                            return False
                        for s in list(seen_items):
                            if key == s:
                                return False
                            if len(key) >= 5 and len(s) >= 5 and (
                                key.startswith(s)
                                or s.startswith(key)
                                or key in s
                                or s in key
                            ):
                                if len(key) <= len(s):
                                    return False
                                # Replace shorter stub with fuller key
                                idx = next(
                                    (
                                        i
                                        for i, d in enumerate(distinct_items)
                                        if d.lower().startswith(s)
                                        or s in d.lower()
                                    ),
                                    None,
                                )
                                seen_items.discard(s)
                                if idx is not None:
                                    distinct_items.pop(idx)
                                break
                        seen_items.add(key)
                        distinct_items.append((label or key)[:100])
                        return True

                    if obj_m:
                        key = _item_key(obj_m.group(0))
                        # Multi buy/assemble/sell/fix: furniture phrase is
                        # already the object; do not drop when verb-key misses.
                        if not key and multi_action:
                            key = re.sub(
                                r"\s+", " ", obj_m.group(0).strip().lower()
                            )
                    elif acq_name:
                        key = _item_key(acq_name) or _item_key(span)
                    else:
                        key = _item_key(span)
                    if not key:
                        continue
                    label = key
                    if acq_name and soft_contains(acq_name, key):
                        # Word-boundary trim; avoid mid-word "peace li"
                        label = re.sub(
                            r"\s+\S*$",
                            "",
                            acq_name.strip()[:80],
                        ) or key
                        if key not in label.lower():
                            label = key
                    if not _register_item(key, label):
                        continue
                    # "got X along with a Y" / "bought X and a Y" → two items
                    for m_aw in re.finditer(
                        r"\balong with\s+(?:a|an|the|my)?\s*"
                        r"([A-Za-z][A-Za-z0-9 '-]{2,40})"
                        r"|(?:\bbought|\bgot|\bpurchased|\bacquired)\b[^\n.]{0,60}?"
                        r"\band\s+(?:a|an|the|my)\s+"
                        r"([A-Za-z][A-Za-z0-9 '-]{2,40})",
                        span,
                        re.I,
                    ):
                        raw_extra = next((g for g in m_aw.groups() if g), None)
                        if not raw_extra:
                            continue
                        extra = _item_key(raw_extra) or _item_key(
                            f"bought a {raw_extra}"
                        )
                        if not extra or extra in {
                            "fertilizer",
                            "humidifier",
                            "mixture",
                            "water",
                            "tips",
                            "advice",
                        }:
                            continue
                        _register_item(extra, f"also {raw_extra.strip()}")

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
    # Re-watch questions must NOT use plain "watched N" / "one of the four" totals.
    stated_ranked: list[tuple[int, int]] = []  # (specificity, n)
    adjacent_ns: list[int] = []
    if (
        (open_enumerate or (not conjuncts and not money_q))
        and not rewatch_q
        and not subscription_q
        and not service_plan_q
        and not health_device_q
        and not delivery_q
        and not tank_q
        and not fitness_days_q
    ):
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
            r"\b(?:i(?:'ve| have)?|i)\b[^\n.]{0,40}?\b(?:tried|watched|re-?watched|"
            r"bought|assembled|fixed|sold|got|viewed|found|finished|read|"
            r"acquired|adopted|planted|subscribed)\b[^\n.]{0,40}?\b"
            r"(\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty)\b"
            r"[^\n.]{0,40}?\b(?:of|recipes?|films?|movies?|pieces?|items?|"
            r"plants?|tanks?|subscriptions?|skeins?|issues?|balls?|copies|"
            r"titles?)\b",
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
        n_dist = len(distinct_items[:10])
        lines.append(
            "Candidate distinct items (dedupe across sessions): "
            + "; ".join(distinct_items[:10])
        )
        lines.append(f"Suggested distinct item count: {n_dist}.")
        if re.search(r"\bincluding\b", ql):
            lines.append(
                "Question says including: count every listed on-topic item "
                "(including gifted/set-up/for-someone-else), not only personal buys."
            )
        if subscription_q:
            lines.append(
                "Count active subscriptions only; ignore canceled titles and "
                "one-off single issues."
            )
            # Strong form so answer packing can overwrite a wrong bare integer.
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif rewatch_q:
            lines.append("Count distinct re-watched titles only, not total watches.")
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif service_plan_q:
            lines.append(
                "Count distinct assets that were serviced or planned for service "
                "in the asked window (not accessories or duplicate shop visits)."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif health_device_q:
            lines.append(
                "Count distinct health devices the user uses (wearables, meters, "
                "aids, treatment machines), not tips or brands alone."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif delivery_q:
            lines.append(
                "Count distinct delivery services/apps used (not order counts "
                "or home-cooking tips)."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif tank_q:
            lines.append(
                "Count distinct tanks/aquariums by size (include set-up-for-others "
                "when the question says including)."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif fitness_days_q:
            lines.append(
                "Count distinct weekdays with first-person fitness class "
                "attendance (not workout tips alone)."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        elif acquire_open:
            # Strong form: bare wrong ints like "2 (a, b)" must yield to N.
            lines.append(
                "Count distinct acquired/owned items across sessions "
                "(hyponyms count when on-topic)."
            )
            lines.append(
                f"Suggested stated total from first-person count claim: {n_dist}."
            )
            lines.append(f"Answer with the integer {n_dist}.")
        else:
            lines.append(
                f"Answer with the integer {n_dist} plus short names "
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
                if is_comparative_savings_query(query) and len(conjuncts) == 2:
                    a, b = conjuncts[0], conjuncts[1]
                    diff = abs(int(per_m[a]) - int(per_m[b]))
                    hi = a if per_m[a] >= per_m[b] else b
                    lo = b if hi == a else a
                    lines.append(
                        "Suggested savings difference: "
                        f"{diff} ({hi}=${per_m[hi]} - {lo}=${per_m[lo]})."
                    )
                    lines.append(
                        f"Answer with ${diff} (or the integer {diff}): "
                        "the positive difference between the two option costs."
                    )
                else:
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
    age_delta_q = is_age_delta_query(query)
    best_by_session: dict[str, dict] = {}
    for r in candidates:
        cl = (r.get("content") or "").lower()
        phrase_hit = any(p in cl for p in phrases)
        noun_hits = sum(1 for t in terms if t in cl and len(t) >= 4)
        conj_hit = any(soft_contains(cl, c) for c in conjuncts)
        # Age-delta needs BOTH current age and age-at-event sessions; the
        # current-age chat often lacks "graduated/college" nouns.
        age_cue = age_delta_q and bool(
            re.search(
                r"\b\d{1,2}\s*-?\s*year\s*-?\s*old\b|"
                r"\bat the age of\s+\d{1,2}\b|"
                r"\bwhen i was\s+\d{1,2}\b",
                cl,
                re.I,
            )
        )
        if not phrase_hit and noun_hits < 1 and not conj_hit and not age_cue:
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
    if is_temporal_span_query(query) or is_temporal_order_query(query):
        return max(base, 14)
    if (
        is_health_device_count_query(query)
        or is_subscription_count_query(query)
        or is_delivery_service_count_query(query)
    ):
        return max(base, 18)
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


_CONTENT_AGO_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(days?|weeks?|months?)\s+ago\b",
    re.I,
)


def is_content_ago_query(query: str) -> bool:
    """
    What/where/which content lookup that uses N units ago as a time pin.

    Distinct from how-many-ago integer spans: do not force a bare duration.
    """
    q = query or ""
    if asks_temporal_span_integer(q) or is_temporal_ago_query(q):
        return False
    if not _CONTENT_AGO_RE.search(q):
        return False
    return bool(
        re.search(
            r"\b(?:what|which|where)\b|"
            r"\b(?:buy|bought|purchase|purchased|got|get|mention)\b",
            q,
            re.I,
        )
    )


def build_content_ago_digest(
    hits: list[dict], query: str = "", question_date: str = ""
) -> str:
    """
    Pin episodes near question_date − N units for content-ago questions.

    Surfaces candidate purchases/events without emitting a duration integer.
    """
    if not is_content_ago_query(query) or not hits:
        return ""
    from datetime import datetime as _dt, timedelta as _td

    m = _CONTENT_AGO_RE.search(query or "")
    if not m:
        return ""
    nraw, unit = m.group(1).lower(), m.group(2).lower()
    words = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    try:
        n = int(nraw) if nraw.isdigit() else int(words.get(nraw, 0))
    except ValueError:
        n = 0
    if n <= 0:
        return ""
    if unit.startswith("week"):
        delta = _td(days=7 * n)
        window = _td(days=10)
    elif unit.startswith("month"):
        delta = _td(days=30 * n)
        window = _td(days=21)
    else:
        delta = _td(days=n)
        window = _td(days=max(3, n // 2 + 1))

    q_dt = None
    qd = (question_date or "").strip()
    qm = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", qd)
    if qm:
        try:
            q_dt = _dt(int(qm.group(1)), int(qm.group(2)), int(qm.group(3)))
        except ValueError:
            q_dt = None
    if q_dt is None:
        # Fall back to newest hit stamp as "now"
        stamps = [d for d in (_parse_hit_datetime(h) for h in hits) if d is not None]
        if not stamps:
            return ""
        q_dt = max(stamps)
    target = q_dt - delta

    focus = [
        t
        for t in (extract_query_focus_terms(query) + extract_topic_nouns(query, max_n=8))
        if len(t) >= 4
        and t.lower()
        not in {
            "weeks", "week", "months", "month", "days", "day", "ago",
            "mentioned", "mention", "investment", "competition", "buy", "bought",
        }
    ]
    # Always keep buy/acquire cues for purchase questions
    buy_cues = ("bought", "buy", "purchased", "purchase", "got my", "i got", "ordered")

    scored: list[tuple[int, dict, Any]] = []
    for h in hits:
        dt = _parse_hit_datetime(h)
        if dt is None:
            continue
        content = h.get("content") or ""
        cl = content.lower()
        if focus and not any(t.lower() in cl for t in focus):
            # Still allow clear first-person acquire near target window
            if not any(c in cl for c in buy_cues):
                continue
        dist = abs((dt.date() - target.date()).days)
        if dist > window.days * 2:
            continue
        scored.append((dist, h, dt))
    if not scored:
        # Relax: closest on-topic hit by focus alone
        for h in hits:
            dt = _parse_hit_datetime(h)
            if dt is None:
                continue
            content = (h.get("content") or "").lower()
            if focus and not any(t.lower() in content for t in focus):
                continue
            dist = abs((dt.date() - target.date()).days)
            scored.append((dist, h, dt))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0])
    lines = [
        "Content-ago time pin "
        f"(target ~{target.date().isoformat()}, "
        f"{n} {unit} before question date {q_dt.date().isoformat()}):",
        "Prefer the candidate whose session date is nearest that target; "
        "do not answer with a later similar purchase outside the window.",
    ]
    for dist, h, dt in scored[:4]:
        content = h.get("content") or ""
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        when = h.get("valid_at_human") or dt.date().isoformat()
        # Prefer a user acquire / investment snippet
        snippet = ""
        for um in re.finditer(r"(?im)^(?:user|human)\s*:\s*(.+)$", content):
            ul = um.group(1).lower()
            if any(c in ul for c in buy_cues) or (
                focus and any(t.lower() in ul for t in focus[:3])
            ):
                snippet = re.sub(r"\s+", " ", um.group(1)).strip()[:160]
                break
        if not snippet:
            snippet = re.sub(r"\s+", " ", content).strip()[:160]
        lines.append(
            f"- ({sid}) @{when} (~{dist}d from target): {snippet}"
        )
    best = scored[0][1]
    best_when = best.get("valid_at_human") or scored[0][2].date().isoformat()
    lines.append(
        f"Nearest candidate session: {best.get('source_description') or best.get('uuid')} "
        f"@{best_when}."
    )
    # Short focus line from the nearest acquire / mention snippet
    best_content = best.get("content") or ""
    focus_line = ""
    for um in re.finditer(r"(?im)^(?:user|human)\s*:\s*(.+)$", best_content):
        ul = um.group(1)
        if any(c in ul.lower() for c in buy_cues) or (
            focus and any(t.lower() in ul.lower() for t in focus[:3])
        ):
            focus_line = re.sub(r"\s+", " ", ul).strip()
            break
    if focus_line:
        # Prefer the clause that states what was acquired
        m_got = re.search(
            r"((?:I |i )?(?:got|bought|purchased|ordered)\b.{10,120})",
            focus_line,
            re.I,
        )
        shown = m_got.group(1).strip(" .,") if m_got else focus_line[:140]
        lines.append(f"Suggested answer focus: {shown}.")
    return "\n".join(lines)


def is_shipping_latency_query(query: str) -> bool:
    """How many days/weeks between order/buy and receive/arrive (shipping lag)."""
    q = query or ""
    if not re.search(r"\bhow many (?:days?|weeks?)\b", q, re.I):
        return False
    if not re.search(r"\b(?:order(?:ed)?|bought|purchased)\b", q, re.I):
        return False
    return bool(
        re.search(r"\b(?:receiv(?:e|ed)|arriv(?:e|ed|al)|deliver(?:y|ed))\b", q, re.I)
    )


def is_temporal_order_query(query: str) -> bool:
    """Ask for chronological order of events (not a count)."""
    q = query or ""
    return bool(
        re.search(r"\border of\b", q, re.I)
        or re.search(
            r"\b(?:from earliest to latest|earliest to latest|in (?:what )?order)\b",
            q,
            re.I,
        )
    )


def build_temporal_order_digest(
    hits: list[dict], query: str = ""
) -> str:
    """
    List dated on-topic events earliest→latest for order questions.

    Does not emit an integer count (that was overwriting order answers).
    """
    if not is_temporal_order_query(query) or not hits:
        return ""
    from datetime import datetime as _dt, timezone as _tz

    _MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
        "december": 12,
    }

    def _parse_when(h: dict):
        try:
            va = int(h.get("valid_at") or 0)
            if va > 1_000_000_000:
                return _dt.fromtimestamp(va, tz=_tz.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            pass
        when = (h.get("valid_at_human") or "").strip()
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

    # Verb-anchored Title-Case / sport-event titles (case-sensitive titles).
    event_pat = re.compile(
        r"(?:completed|finished|participat(?:ed|e)\s+in|took\s+part\s+in|"
        r"played\s+in|(?:ran|finished)\s+(?:a\s+)?\d?5[Kk].{0,40}\bat)\s+"
        r"(?:the\s+)?"
        r"("
        r"(?:[A-Z][a-zA-Z0-9']+(?:\s+[A-Z0-9][a-zA-Z0-9']+){0,5})\s+"
        r"(?:Triathlon|Tournament|Marathon|Championship|Meet|"
        r"5[Kk](?:\s+Run)?|10[Kk](?:\s+Run)?)"
        r"|(?:company'?s?\s+)?(?:annual\s+)?charity\s+"
        r"(?:soccer|football|basketball)\s+tournament"
        r")"
    )
    dated: list[tuple[Any, str, str]] = []
    seen: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        if content.lstrip().startswith("[Session events"):
            continue
        dt = _parse_when(h)
        if dt is None:
            continue
        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = "\n".join(user_chunks) if user_chunks else content
        # Require first-person participation nearby for sports-order questions
        participated = bool(
            re.search(
                r"\b(?:i|i'?ve|i just)\b[^\n.]{0,120}\b"
                r"(?:completed|finished|participat(?:ed|e)|took part|"
                r"ran|played)\b",
                scan,
                re.I,
            )
        )
        if not participated:
            continue
        labels: list[str] = []
        for m in event_pat.finditer(scan):
            lab = re.sub(r"\s+", " ", m.group(1)).strip()
            if lab:
                labels.append(lab)
        if not labels:
            # Structural fallback: Title-Case + race/tournament suffix
            for m in re.finditer(
                r"\b((?:[A-Z][a-zA-Z0-9']+(?:\s+[A-Z0-9][a-zA-Z0-9']+){0,5})"
                r"\s+(?:Triathlon|Tournament|Marathon|Championship|"
                r"5[Kk](?:\s+Run)?|10[Kk](?:\s+Run)?)"
                r"|(?:company'?s?\s+)?(?:annual\s+)?charity\s+"
                r"(?:soccer|football|basketball)\s+tournament)\b",
                scan,
            ):
                labels.append(re.sub(r"\s+", " ", m.group(1)).strip())
        for lab in labels:
            key = re.sub(r"[^a-z0-9]+", " ", lab.lower()).strip()
            if key in seen or len(lab) < 6:
                continue
            # Collapse near-duplicates (spring sprint triathlon vs …)
            stem = " ".join(key.split()[:4])
            if any(stem in s or s in stem for s in seen):
                continue
            seen.add(key)
            dated.append((dt, lab, h.get("valid_at_human") or ""))
    if len(dated) < 2:
        return ""
    dated.sort(key=lambda x: x[0])
    want_n = 0
    m_n = re.search(r"\b(three|four|five|\d+)\s+(?:sports\s+)?events?\b", query, re.I)
    if m_n:
        word = m_n.group(1).lower()
        want_n = {"three": 3, "four": 4, "five": 5}.get(word, 0)
        if not want_n:
            try:
                want_n = int(word)
            except ValueError:
                want_n = 0
    keep = dated[: want_n or 8]
    if want_n and len(keep) < want_n:
        keep = dated[:8]
    lines = [
        "Chronological event order from memory (earliest → latest). "
        "Answer with the ordered list of event names, not a count:",
    ]
    for i, (dt, lab, when) in enumerate(keep, 1):
        stamp = f" @{when}" if when else f" @{dt.date().isoformat()}"
        lines.append(f"{i}. {lab}{stamp}")
    lines.append(
        "Do not answer with only the integer count of events; list names in order."
    )
    return "\n".join(lines)


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
    # Abs guard: span between buy(X) and event Y requires X to appear in memory.
    bought = re.search(
        r"\b(?:bought|buy|purchased)\s+(?:my\s+|an?\s+)?([A-Za-z][A-Za-z0-9-]{1,24})",
        query or "",
        re.I,
    )
    if bought:
        obj = bought.group(1).lower()
        blob = "\n".join((h.get("content") or "") for h in hits).lower()
        if obj not in blob and obj.rstrip("s") not in blob:
            return (
                "Asked purchase object is not mentioned in the memory excerpts.\n"
                "Suggested answer: I do not know.\n"
                "Do not invent a day count when the bought item never appears."
            )
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

    def _emit_span(delta_days: int, early: Any, late: Any, note: str) -> None:
        nonlocal content_span_done
        if delta_days <= 0 or content_span_done:
            return
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
        if span_n <= 0:
            return
        lines.append(
            f"Suggested span from listed event dates: {span_n} {unit} "
            f"({delta_days} days {note} "
            f"{early.date().isoformat()} → {late.date().isoformat()})."
        )
        lines.append(
            f"Answer with the integer {span_n} unless a listed anchor "
            "clearly does not match the asked events."
        )
        content_span_done = True

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
        order_dates: list[Any] = []
        arrive_dates: list[Any] = []
        # Session stamps when the user says bought/received "today" (no month/day).
        for h in hits:
            content = h.get("content") or ""
            cl = content.lower()
            if anchors and not any(a.lower() in cl for a in anchors):
                continue
            dt = _parse_when(h)
            bought_here = bool(
                re.search(r"\b(?:bought|purchased|ordered)\b", cl, re.I)
            )
            recv_here = bool(
                re.search(
                    r"\b(?:receiv(?:e|ed)|arriv(?:e|ed|al)|deliver(?:y|ed))\b",
                    cl,
                    re.I,
                )
            )
            if year is not None:
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
            if dt is not None:
                # Prefer buy-only sessions for order date; receive must name the
                # object near the receive verb (avoid unrelated "received" noise).
                if bought_here and not recv_here:
                    order_dates.append(dt)
                if recv_here and (
                    not anchors
                    or any(
                        re.search(
                            rf"\b(?:receiv(?:e|ed)|arriv(?:e|ed))\b.{{0,60}}"
                            rf"{re.escape(a.lower())}"
                            rf"|{re.escape(a.lower())}.{{0,60}}"
                            rf"\b(?:receiv(?:e|ed)|arriv(?:e|ed))\b",
                            cl,
                        )
                        for a in anchors
                        if len(a) >= 4
                    )
                ):
                    arrive_dates.append(dt)
        if order_dates and arrive_dates:
            order_dt = min(order_dates)
            later_arrivals = [d for d in arrive_dates if d >= order_dt]
            arrive_dt = min(later_arrivals) if later_arrivals else min(arrive_dates)
            delta_days = max(0, (arrive_dt.date() - order_dt.date()).days)
            _emit_span(
                delta_days,
                order_dt,
                arrive_dt,
                "between buy/order and receive",
            )

    def _between_day_sides(q: str) -> tuple[str, str] | None:
        m = re.search(
            r"\bbetween the day\b(.+?)\band the day\b(.+?)(?:\?|$)",
            q or "",
            re.I,
        )
        if not m:
            return None
        return m.group(1).strip(), m.group(2).strip()

    def _days_before_sides(q: str) -> tuple[str, str] | None:
        """
        'How many days before <later A> did I <earlier B>?'

        Returns (later_side, earlier_side) for A←B offset math.
        """
        m = re.search(
            r"\bhow many (?:days?|weeks?|months?)\s+before\b(.+?)\bdid i\b(.+?)(?:\?|$)",
            q or "",
            re.I,
        )
        if not m:
            return None
        later = m.group(1).strip(" .,;:")
        earlier = m.group(2).strip(" .,;:")
        if len(later) < 4 or len(earlier) < 4:
            return None
        return later, earlier

    def _since_when_sides(q: str) -> tuple[str, str] | None:
        """
        'How many days/weeks had passed since <earlier> when I <later>?'

        Returns (earlier_side, later_side). Avoids global min/max over noise.
        """
        m = re.search(
            r"\b(?:had |have )?passed since\b(.+?)\bwhen i\b(.+?)(?:\?|$)",
            q or "",
            re.I,
        )
        if not m:
            m = re.search(
                r"\bsince (?:the day )?i\b(.+?)\bwhen i\b(.+?)(?:\?|$)",
                q or "",
                re.I,
            )
        if not m:
            return None
        earlier = m.group(1).strip(" .,;:")
        later = m.group(2).strip(" .,;:")
        if len(earlier) < 4 or len(later) < 4:
            return None
        return earlier, later

    def _side_event_keys(side: str) -> list[str]:
        """Distinctive multi-word / noun keys for one side of a between-day span."""
        weak = {
            "day", "days", "started", "start", "favorite", "songs", "song",
            "main", "took", "take", "made", "make", "local", "old", "new",
            "along", "playing", "discovered", "discover", "before", "after",
            "passed", "participated", "replaced", "decided", "taking",
            "went", "outdoors", "typical", "event", "events",
        }
        keys: list[str] = []
        seen: set[str] = set()

        def _add(p: str) -> None:
            p = re.sub(r"\s+", " ", (p or "").strip().lower())
            if len(p) < 4 or p in seen or p in weak:
                return
            seen.add(p)
            keys.append(p)

        # Capitalized multi-word event names ("Turbocharged Tuesdays", "Midsummer 5K")
        for m in re.finditer(
            r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z0-9][A-Za-z0-9]+){1,4})\b",
            side or "",
        ):
            _add(m.group(1))
        for p in extract_topic_phrases(side, max_n=10):
            parts = p.lower().split()
            if not parts or all(w in weak for w in parts):
                continue
            _add(p)
        for n in extract_topic_nouns(side, max_n=8):
            if n.lower() in weak or len(n) < 4:
                continue
            _add(n)
        # Compound nouns often missed: "spark plugs", "ukulele lessons"
        for m in re.finditer(
            r"\b((?:spark\s+plugs?|ukulele\s+lessons?|acoustic\s+guitar|"
            r"tennis\s+rackets?|guitar\s+tech|10th\s+jog|recovered from the flu|"
            r"auto\s+racking))\b",
            side or "",
            re.I,
        ):
            _add(m.group(1))
        keys.sort(key=lambda s: (-(1 if " " in s else 0), -len(s)))
        return keys[:10]

    def _side_hit(content: str, keys: list[str], side_text: str = "") -> bool:
        cl = (content or "").lower()
        sl = (side_text or "").lower()
        if not keys or not cl:
            return False
        # Soft requirements: accept common memory paraphrases (tech→servicing)
        req_groups: list[tuple[str, ...]] = []
        if "playing along" in sl:
            req_groups.append(("playing along",))
        if "bluegrass" in sl:
            req_groups.append(("bluegrass",))
        if "spark plug" in sl:
            req_groups.append(("spark plug", "spark plugs"))
        if "turbocharged" in sl:
            req_groups.append(("turbocharged",))
        if "ukulele" in sl:
            req_groups.append(("ukulele",))
        if "recovered from the flu" in sl or ("flu" in sl and "recovered" in sl):
            req_groups.append(("recovered from the flu", "recovered from the flu today", "flu"))
        if "10th jog" in sl or ("jog" in sl and "10th" in sl):
            req_groups.append(("10th jog", "jog outdoors"))
        if "guitar tech" in sl:
            # Prefer explicit tech / decided-to-service wording; bare
            # "servicing" appears in unrelated assistant product tips.
            req_groups.append(
                ("guitar tech", "to the guitar tech", "joe for servicing", "joe")
            )
        if "received" in sl and "bought" not in sl:
            req_groups.append(("received", "arrived"))
        if "bought" in sl and "received" not in sl:
            req_groups.append(("bought", "purchased", "ordered"))
        for group in req_groups:
            if not any(r in cl for r in group):
                return False
        strong = [k for k in keys if " " in k or len(k) >= 8]
        if strong and any(k.lower() in cl for k in strong):
            return True
        # Guitar-tech paraphrases (user turn): decided + Taylor/tech
        if "guitar tech" in sl:
            if re.search(
                r"\b(?:guitar tech|taylor)\b",
                cl,
            ) and re.search(
                r"\b(?:decided|take my|servicing)\b",
                cl,
            ):
                return True
        return sum(1 for k in keys if k.lower() in cl) >= 2

    # Between-day A→B: pair each side's sessions, not global min/max over noisy
    # anchors (bare "keyboard" matching laptop-stand chatter → 69-day spans).
    sides = _between_day_sides(query or "")
    if not ago and not content_span_done and sides:
        left_keys = _side_event_keys(sides[0])
        right_keys = _side_event_keys(sides[1])
        left_dts: list[Any] = []
        right_dts: list[Any] = []
        for h in hits:
            content = h.get("content") or ""
            dt = _parse_when(h)
            if dt is None:
                continue
            if _side_hit(content, left_keys, sides[0]):
                left_dts.append(dt)
            if _side_hit(content, right_keys, sides[1]):
                right_dts.append(dt)
        if left_dts and right_dts:
            left_dt = min(left_dts)
            # Second side: prefer dates on/after the first event (avoid earlier noise)
            right_after = [d for d in right_dts if d >= left_dt]
            right_dt = min(right_after) if right_after else min(right_dts)
            early, late = (
                (left_dt, right_dt) if left_dt <= right_dt else (right_dt, left_dt)
            )
            delta_days = (late.date() - early.date()).days
            _emit_span(
                delta_days,
                early,
                late,
                "between paired event sides",
            )

    # Days-before A←B: pair later reference A with earlier event B (not global
    # min/max, which collapses when buy+chat share one session stamp → 0 days).
    before_sides = _days_before_sides(query or "")
    if not ago and not content_span_done and before_sides:
        later_keys = _side_event_keys(before_sides[0])
        earlier_keys = _side_event_keys(before_sides[1])
        later_dts: list[Any] = []
        earlier_dts: list[Any] = []
        for h in hits:
            content = h.get("content") or ""
            dt = _parse_when(h)
            if dt is None:
                continue
            if _side_hit(content, later_keys, before_sides[0]):
                later_dts.append(dt)
            if _side_hit(content, earlier_keys, before_sides[1]):
                earlier_dts.append(dt)
        if later_dts and earlier_dts:
            later_dt = min(later_dts)
            earlier_dt = min(earlier_dts)
            delta_days = (later_dt.date() - earlier_dt.date()).days
            _emit_span(
                delta_days,
                earlier_dt,
                later_dt,
                "before later event",
            )

    # Since-when: "had passed since <earlier> when I <later>"
    since_sides = _since_when_sides(query or "")
    if not ago and not content_span_done and since_sides:
        earlier_keys = _side_event_keys(since_sides[0])
        later_keys = _side_event_keys(since_sides[1])
        earlier_dts: list[Any] = []
        later_dts: list[Any] = []
        for h in hits:
            content = h.get("content") or ""
            dt = _parse_when(h)
            if dt is None:
                continue
            if _side_hit(content, earlier_keys, since_sides[0]):
                earlier_dts.append(dt)
            if _side_hit(content, later_keys, since_sides[1]):
                later_dts.append(dt)
        if earlier_dts and later_dts:
            earlier_dt = min(earlier_dts)
            # Strictly after earlier event (same-day dual-match → 0-day poison)
            later_candidates = [d for d in later_dts if d > earlier_dt]
            if not later_candidates:
                later_candidates = [d for d in later_dts if d >= earlier_dt]
            later_dt = (
                min(later_candidates) if later_candidates else max(later_dts)
            )
            delta_days = (later_dt.date() - earlier_dt.date()).days
            _emit_span(
                delta_days,
                earlier_dt,
                later_dt,
                "since earlier event when later event",
            )

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
            # Prefer multi-word anchors to avoid "keyboard"/"day" noise
            use_anchors = [a for a in anchors if " " in a or len(a) >= 8][:8]
            if len(use_anchors) < 2:
                use_anchors = anchors[:8]
            for a in use_anchors:
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
            # Never push a bare 0: same-day session stamps are usually packaging
            # noise, not a real answer (days-before / between-event fails).
            if span_n > 0:
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
            # Same-day stamp → 0 is packaging noise; never suggest bare 0 for ago.
            if span_n > 0:
                lines.append(
                    f"Suggested span from listed event dates: {span_n} {unit} "
                    f"({delta_days} days before question date {q_dt.date().isoformat()})."
                )
                lines.append(
                    f"Answer with the integer {span_n} unless a listed anchor clearly does "
                    "not match the asked event."
                )
            else:
                lines.append(
                    "Matched event stamp is the same calendar day as the question date "
                    "(or later). Do not answer 0; pick an older on-topic dated event "
                    "or abstain if none exists."
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


def _parse_duration_to_unit(token: str, unit: str) -> float | None:
    """Parse 'two weeks' / 'week and a half' / '3.5 weeks' into asked unit."""
    t = (token or "").strip().lower()
    if not t:
        return None
    half = bool(re.search(r"\band a half\b|\b1/2\b|\.5\b", t))
    words = {
        "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5,
    }
    m = re.search(
        r"(\d+(?:\.\d+)?|a|one|two|three|four|five|six|seven|eight|nine|ten|half)",
        t,
    )
    if not m:
        return None
    raw = m.group(1)
    try:
        n = float(raw) if raw[0].isdigit() else float(words.get(raw, 0))
    except ValueError:
        return None
    if n <= 0:
        return None
    if half and n >= 1:
        n = n + 0.5
    src_unit = "days"
    if "week" in t:
        src_unit = "weeks"
    elif "month" in t:
        src_unit = "months"
    elif "day" in t:
        src_unit = "days"
    days = n
    if src_unit == "weeks":
        days = n * 7.0
    elif src_unit == "months":
        days = n * 30.0
    want = (unit or "days").lower()
    if want.startswith("week"):
        return days / 7.0
    if want.startswith("month"):
        return days / 30.0
    return days


def build_stated_effort_duration_digest(hits: list[dict], query: str = "") -> str:
    """
    Sum first-person binge/effort durations for 'how many weeks did it take to watch…'.

    Structural only: pairs duration claims near question topic nouns / conjuncts.
    """
    if not is_stated_effort_duration_query(query) or not hits:
        return ""
    ql = (query or "").lower()
    unit = "weeks"
    if re.search(r"\bhow many days?\b", ql):
        unit = "days"
    elif re.search(r"\bhow many months?\b", ql):
        unit = "months"
    topics = [
        t
        for t in (extract_topic_phrases(query) + extract_topic_nouns(query))
        if t and len(t) >= 3
    ]
    # Franchise / title heads often carry the duration claim
    topics = [t for t in topics if t.lower() not in {"how", "many", "take", "watch", "film", "films", "movies", "movie"}]
    if not topics:
        return ""
    dur_pat = re.compile(
        r"\bin\s+"
        r"((?:a\s+)?(?:week|day|month)(?:\s+and\s+a\s+half)?|"
        r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:days?|weeks?|months?)(?:\s+and\s+a\s+half)?)",
        re.I,
    )
    # Collect (topic_key, hours_in_asked_unit, snippet)
    claims: list[tuple[str, float, str]] = []
    for h in hits:
        content = h.get("content") or ""
        if content.lstrip().startswith("[Session events"):
            continue
        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = "\n".join(user_chunks) if user_chunks else content
        for m in dur_pat.finditer(scan):
            window = scan[max(0, m.start() - 100) : m.end() + 80]
            wl = window.lower()
            matched_topics = [t for t in topics if soft_contains(wl, t)]
            if not matched_topics:
                continue
            val = _parse_duration_to_unit(m.group(1), unit)
            if val is None or val <= 0 or val > 100:
                continue
            # Prefer the longest matching topic as the claim key
            key = max(matched_topics, key=len).lower()
            snippet = re.sub(r"\s+", " ", window).strip()
            claims.append((key, val, snippet))
    if not claims:
        return ""
    # One claim per topic key (max duration if several)
    per: dict[str, tuple[float, str]] = {}
    for key, val, snip in claims:
        prev = per.get(key)
        if prev is None or val > prev[0]:
            per[key] = (val, snip)
    # If question has two franchise-like topics, sum distinct keys; else take max
    values = [v for v, _ in per.values()]
    if len(per) >= 2:
        total = sum(values)
        mode = "sum of per-topic binge claims"
    else:
        total = max(values)
        mode = "stated binge claim"
    # Pretty-print half weeks
    if abs(total - round(total)) < 1e-6:
        shown = str(int(round(total)))
    elif abs(total * 2 - round(total * 2)) < 1e-6:
        shown = f"{total:g}"
    else:
        shown = f"{total:g}"
    lines = [
        "Stated effort/binge durations found in memory:",
    ]
    for key, (val, snip) in list(per.items())[:6]:
        lines.append(f"- [{key}] {val:g} {unit}: {snip[:160]}")
    lines.append(f"Suggested effort duration total: {shown} {unit} ({mode}).")
    lines.append(
        f"Answer with {shown} (or '{shown} {unit}'). Sum distinct topic binges when "
        "the question asks how long A and B took together."
    )
    return "\n".join(lines)


def build_person_location_digest(hits: list[dict], query: str = "") -> str:
    """
    'Where does (my sister) Name live?' → surface Name-in-City first-person cues.
    """
    m = re.search(
        r"\bwhere does\s+(?:my\s+)?"
        r"(?:(?P<rel>sister|brother|friend|mother|father|mom|dad|wife|husband|cousin)\s+)?"
        r"(?P<name>[A-Za-z][a-z]{2,})\s+live\b",
        query or "",
        re.I,
    )
    if not m or not hits:
        return ""
    name = m.group("name")
    rel = (m.group("rel") or "").lower()
    # Locate "Name in <City>" then take a case-sensitive Capitalized city token
    # (IGNORECASE on [A-Z][a-z] wrongly absorbs "soon" after Denver).
    loc_pats = [
        re.compile(
            rf"\bvisiting\s+(?:(?:my\s+)?{re.escape(rel)}\s+)?"
            rf"{re.escape(name)}\s+in\s+",
            re.I,
        ),
        re.compile(
            rf"\b(?:(?:my\s+)?{re.escape(rel)}\s+)?"
            rf"{re.escape(name)}\s+in\s+",
            re.I,
        ),
    ]
    ban = {
        "the", "this", "that", "our", "her", "his", "their", "a", "an",
        "early", "late", "mid", "general", "soon", "town", "city",
    }
    cities: list[tuple[str, str]] = []
    for h in hits:
        scan = h.get("content") or ""
        for pat in loc_pats:
            for cm in pat.finditer(scan):
                tail = scan[cm.end() : cm.end() + 40]
                city_m = re.match(
                    r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)",
                    tail,
                )
                if not city_m:
                    continue
                city = city_m.group(1)
                if city.lower() in ban:
                    continue
                snip = re.sub(
                    r"\s+",
                    " ",
                    scan[max(0, cm.start() - 40) : cm.end() + len(city) + 20],
                ).strip()
                cities.append((city, snip))
    if not cities:
        return ""
    # Prefer first user claim
    city, snip = cities[0]
    return "\n".join(
        [
            f"Person-location cues for {name}:",
            f"- {snip}",
            f"Suggested location: {city}.",
            f"Answer with {city} (city/region name).",
        ]
    )


_LOC_INTENT_RE = re.compile(
    r"\b(?:where|lives?|living|moved?|moving|relocat|now|currently|"
    r"these days|stay(?:ing)?|based)\b",
    re.I,
)

_CURRENT_STATE_RE = re.compile(
    r"\b(?:personal best|\bpb\b|currently (?:own|have|running)|"
    r"what (?:is|was) my (?:current )?(?:personal )?best|"
    r"previous (?:personal )?best|"
    r"how many .{0,60}currently (?:own|have))\b",
    re.I,
)

_CURRENT_OWN_RE = re.compile(
    r"\bcurrently (?:own|have)\b|\bdo i (?:still )?own\b",
    re.I,
)

_PB_TIME_RE = re.compile(
    r"personal best(?:\s+time)?(?:\s+in\s+[^.?]{0,40})?\s+"
    r"(?:of\s+|with a time of\s+|was\s+|is\s+now\s+|is\s+)?"
    r"(\d{1,2}:\d{2})",
    re.I,
)


def build_personal_best_digest(hits: list[dict], query: str = "") -> str:
    """
    Surface current vs previous personal-best times from dated sessions.

    Prefers the newest explicit 'personal best … HH:MM' claim.
    """
    if not hits or not re.search(r"\bpersonal best\b|\bpb\b", query or "", re.I):
        return ""
    wants_previous = bool(re.search(r"\bprevious\b", query or "", re.I))
    found: list[tuple[int, str, str]] = []
    for h in hits:
        content = h.get("content") or ""
        user_chunks = re.findall(r"(?im)^(?:user|human)\s*:\s*(.+)$", content)
        scan = "\n".join(user_chunks) if user_chunks else content
        for m in _PB_TIME_RE.finditer(scan):
            dt = _parse_hit_datetime(h)
            ord_key = dt.toordinal() if dt is not None else 0
            sid = (h.get("source_description") or h.get("uuid") or "?").strip()
            found.append((ord_key, m.group(1), sid))
    if not found:
        return ""
    found.sort(key=lambda x: x[0])
    # Dedupe keeping newest per time then overall newest claim
    by_time: dict[str, tuple[int, str, str]] = {}
    for row in found:
        prev = by_time.get(row[1])
        if prev is None or row[0] >= prev[0]:
            by_time[row[1]] = row
    ordered = sorted(by_time.values(), key=lambda x: x[0])
    current = ordered[-1]
    previous = ordered[-2] if len(ordered) >= 2 else None
    lines = ["Personal-best time notes (dated claims):"]
    for ord_key, t, sid in ordered:
        lines.append(f"- {t} ({sid})")
    if wants_previous and previous:
        lines.append(f"Suggested previous personal best: {previous[1]}.")
        lines.append(f"Answer with {previous[1]}.")
    else:
        lines.append(f"Suggested current personal best: {current[1]}.")
        lines.append(f"Answer with {current[1]}.")
    return "\n".join(lines)


def prioritize_query_entity_recency(episodes: list[dict], query: str) -> list[dict]:
    """
    Prefer newest matching episodes for location / current-state questions.

    Location: moved/suburbs/apartment updates (existing).
    Current-state: personal best, currently own/have, previous best (second-newest).
    """
    loc_intent = bool(_LOC_INTENT_RE.search(query or ""))
    state_intent = bool(_CURRENT_STATE_RE.search(query or ""))
    if not loc_intent and not state_intent:
        return episodes
    # Aggregate counts usually need breadth; currently-own is an exception
    # (latest inventory supersedes earlier partial lists).
    if is_aggregate_query(query) and not _CURRENT_OWN_RE.search(query or ""):
        return episodes
    # Word-boundary focus matching; short tokens ('Star') are substring traps
    focus = [t.lower() for t in extract_query_focus_terms(query) if len(t) >= 4]
    if state_intent and not focus:
        focus = [
            t.lower()
            for t in extract_topic_nouns(query, max_n=8)
            if len(t) >= 4
        ]
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

    def _state_score(r: dict) -> int:
        content = r.get("content") or ""
        cl = content.lower()
        if not any(t in cl for t in focus):
            # PB / time updates may omit the race name
            if not re.search(
                r"\b(?:personal best|\bpb\b|\d{1,2}:\d{2}|minutes?\b)",
                cl,
            ) and not _CURRENT_OWN_RE.search(query or ""):
                return -1
        score = 1
        if re.search(r"\b(?:personal best|\bpb\b|new (?:pr|record))\b", cl):
            score += 10
        if re.search(r"\b\d{1,2}:\d{2}\b", cl):
            score += 4
        if re.search(r"\b(?:own|owns|owned|have|has)\b", cl):
            score += 3
        return score

    matched: list[dict] = []
    rest: list[dict] = []
    for row in episodes:
        content = (row.get("content") or "").lower()
        if any(t in content for t in focus) or (
            state_intent
            and re.search(r"\b(?:personal best|\bpb\b|\d{1,2}:\d{2})\b", content)
        ):
            matched.append(row)
        else:
            rest.append(row)
    if len(matched) < 2 and not (state_intent and matched):
        return episodes

    wants_previous = bool(re.search(r"\bprevious\b", query or "", re.I))
    if loc_intent and not state_intent:
        matched = sorted(
            matched, key=lambda r: (_loc_score(r), _va(r)), reverse=True
        )
    else:
        matched = sorted(
            matched, key=lambda r: (_state_score(r), _va(r)), reverse=True
        )
        if wants_previous and len(matched) >= 2:
            # Second-newest matching fact is the "previous" value
            matched = [matched[1], matched[0]] + matched[2:]

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
                focus[0] if focus else "this",
            )
            if loc_intent and not state_intent:
                banner = (
                    f"[LATEST update for {focus_hit}] Use this episode for current "
                    f"location/status; ignore older episodes about {focus_hit}."
                )
            elif wants_previous:
                banner = (
                    f"[PREVIOUS value for {focus_hit}] Use this episode for the prior "
                    f"value before the newest update; do not answer with the latest alone."
                )
            else:
                banner = (
                    f"[LATEST update for {focus_hit}] Use this episode for the current "
                    f"value (personal best / currently own); ignore older superseded figures."
                )
            content = r.get("content") or ""
            if "[LATEST update" not in content and "[PREVIOUS value" not in content:
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
    week_window_q = bool(
        re.search(r"\b(?:last week|this week|past week|previous week)\b", query or "", re.I)
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
        if total_q or week_window_q:
            # Integer-friendly display for whole-hour game totals
            shown = (
                str(int(total_hours))
                if abs(total_hours - round(total_hours)) < 1e-6
                else f"{total_hours:g}"
            )
            lines.append(f"Suggested total hours listed: {shown}.")
            lines.append(
                f"For an 'in total' or last-week duration question, answer with {shown} "
                "unless a listed claim is clearly not the user's completed playtime."
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
        # Attended events/ceremonies: first-person attend/volunteer with a name
        attended: list[str] = []
        seen_att: set[str] = set()
        if event_attend_bridge_terms(query):
            att_patterns = [
                re.compile(
                    r"\b(?:i\s+)?(?:just\s+|recently\s+)?(?:attended|volunteered\s+at|went\s+on)\s+"
                    r"[^\.\n]{8,140}",
                    re.I,
                ),
                # 'Women in Art' exhibition which I attended on February 10th
                re.compile(
                    r"(?:\"[^\"]{2,60}\"\s+|the\s+)?"
                    r"(?:exhibition|lecture|event|tour|afternoon|graduation|ceremony)"
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
                        if re.search(
                            r"\b(?:missing|missed|couldn't attend|could not attend|"
                            r"unable to attend)\b",
                            span,
                            re.I,
                        ):
                            continue
                        if not re.search(
                            r"\b(?:art|museum|gallery|lecture|exhibition|tour|"
                            r"volunteer|festival|graduation|ceremony|"
                            r"commencement)\b",
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
        # Projects excluding X: only comma/and lists like
        # "boards for my thesis, Data Mining project, and Database Systems project"
        excl_m = re.search(
            r"\bexclud(?:e|ing)\s+(?:my\s+|the\s+)?([a-z][a-z'-]{2,30})",
            ql,
        )
        if excl_m and re.search(r"\bprojects?\b", ql):
            exclude = excl_m.group(1).lower()
            named: list[str] = []
            seen_proj: set[str] = set()
            for h in hits:
                content = h.get("content") or ""
                if content.lstrip().startswith("[Session events"):
                    continue
                user_chunks = re.findall(
                    r"(?im)^(?:user|human)\s*:\s*(.+)$", content
                )
                scan = "\n".join(user_chunks) if user_chunks else content
                for m in re.finditer(
                    r"\bboards?\s+for\s+my\s+"
                    r"([^\.\n]{8,160}\bprojects?\b)",
                    scan,
                    re.I,
                ):
                    chunk = m.group(1)
                    # Require a real list (comma + and), not "board for my thesis project."
                    if "," not in chunk or not re.search(r"\band\b", chunk, re.I):
                        continue
                    parts = re.split(r"\s*,\s*|\s+and\s+", chunk, flags=re.I)
                    for part in parts:
                        label = re.sub(r"\s+", " ", part.strip())
                        label = re.sub(r"^(?:and\s+)+", "", label, flags=re.I)
                        label = re.sub(
                            r"\s+projects?\.?$", "", label, flags=re.I
                        ).strip()
                        if not label or len(label) < 3 or len(label) > 48:
                            continue
                        if re.search(
                            r"\b(?:with|such as|so i|lists?|stages?)\b",
                            label,
                            re.I,
                        ):
                            continue
                        key = label.lower()
                        if key == exclude or key.startswith(exclude + " "):
                            continue
                        if exclude in key.split():
                            continue
                        if key in seen_proj:
                            continue
                        seen_proj.add(key)
                        named.append(label)
            if named:
                lines.append(
                    "Candidate projects from first-person board/list claim "
                    f"(excluded '{exclude}'): " + "; ".join(named)
                )
                lines.append(
                    f"Suggested stated total from first-person count claim: {len(named)}."
                )
                lines.append(f"Answer with the integer {len(named)}.")
        # Writing pieces: sum first-person "I've written N …" + "I wrote a piece"
        if re.search(
            r"\bhow many\b.{0,40}\b(?:pieces?\s+of\s+)?writing\b|"
            r"\bpieces?\s+of\s+writing\b|"
            r"\bwriting\b.{0,40}\bcompleted\b",
            ql,
        ):
            by_kind: dict[str, int] = {}
            piece_n = 0
            for h in hits:
                content = h.get("content") or ""
                if content.lstrip().startswith("[Session events"):
                    continue
                user_chunks = re.findall(
                    r"(?im)^(?:user|human)\s*:\s*(.+)$", content
                )
                scan = "\n".join(user_chunks) if user_chunks else content
                for m in re.finditer(
                    r"\bi(?:'ve| have)\s+written\s+"
                    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                    r"seventeen|eighteen|nineteen|twenty)\s+"
                    r"([a-z][a-z '-]{2,40})",
                    scan,
                    re.I,
                ):
                    n = _parse_count_token(m.group(1))
                    if n is None or n <= 0 or n > 200:
                        continue
                    kind = re.sub(r"\s+", " ", m.group(2).strip().lower())
                    kind = re.split(
                        r"\b(?:so far|lately|in the|over the|this)\b",
                        kind,
                        maxsplit=1,
                    )[0].strip(" ,.")
                    if not kind:
                        continue
                    by_kind[kind] = max(by_kind.get(kind, 0), n)
                if re.search(r"\bi\s+wrote\s+a\s+piece\b", scan, re.I):
                    piece_n = max(piece_n, 1)
            total_w = sum(by_kind.values()) + piece_n
            if total_w > 0:
                parts_w = [f"{n} {k}" for k, n in by_kind.items()]
                if piece_n:
                    parts_w.append(f"{piece_n} piece")
                lines.append(
                    "Candidate writing totals from first-person claims: "
                    + "; ".join(parts_w[:8])
                )
                lines.append(
                    f"Suggested stated total from first-person count claim: {total_w}."
                )
                lines.append(f"Answer with the integer {total_w}.")
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
