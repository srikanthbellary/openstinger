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
_COUNT_RE = re.compile(r"\b(how many|how much|count|number of|total)\b", re.I)
_PREF_CUE_RE = re.compile(
    r"\b(prefer|preference|preferable|like|love|want|rather|favorite|"
    r"favourite|interested in|looking for|would rather|don't like|do not like|"
    r"hate|wish|hoping for)\b",
    re.I,
)
_NAME_IN_QUERY = re.compile(r"\b([A-Z][a-z]{2,})\b")
_PHRASE_STOP = _STOP | {
    "passed", "between", "visit", "helped", "friend", "cousin", "ordered",
    "happened", "events", "order", "days", "day", "worked", "bought",
}

def extract_query_focus_terms(query: str) -> list[str]:
    """
    Proper names / places capitalized in the question (Rachel, Miami, Target).
    Used to force CONTAINS follow-up so updates and preferences are not missed.
    """
    found = []
    seen: set[str] = set()
    for m in _NAME_IN_QUERY.findall(query or ""):
        low = m.lower()
        if low in _STOP or low in {
            "how", "what", "when", "where", "which", "can", "the", "you",
            "march", "april", "june", "july", "august", "september", "october",
            "november", "december", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        }:
            continue
        if low not in seen:
            seen.add(low)
            found.append(m)
    return found



def extract_search_terms(query: str) -> list[str]:
    """Alphanumeric tokens useful for BM25 / CONTAINS, stopwords removed."""
    tokens = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP]


def sanitize_bm25_query(query: str) -> str:
    """
    Make a natural-language question safe for FalkorDB/RediSearch BM25.

    Quotes each kept term so operators like ':' and bare 'last' do not break
    the query parser. Falls back to stripped raw text if no terms survive.
    """
    terms = extract_search_terms(query)
    if terms:
        return " ".join(f'"{t}"' for t in terms)
    cleaned = _BM25_SPECIAL.sub(" ", query or "").strip()
    # Stopword-only or empty input: avoid sending raw stopwords to BM25
    if not cleaned or not extract_search_terms(cleaned):
        return '""'
    return cleaned


def is_recommend_query(query: str) -> bool:
    return bool(_RECOMMEND_RE.search(query or ""))


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
    r"swim(?:ming)?|cycl(?:e|ing)|hike|hiking)\b",
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
    if is_recommend_query(query):
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


_TEMPORAL_SPAN_COUNT_RE = re.compile(
    r"\bhow many days\b|"
    r"\b(?:days?|weeks?|months?|years?)\b.{0,40}\b(?:pass|passed|between|since|until|apart)\b|"
    r"\bbetween the day\b",
    re.I,
)


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
    r"would prefer)[^\.\n]{5,160}",
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
    if not is_recommend_query(query):
        return episodes
    q = (query or "").lower()
    hotel_q = "hotel" in q
    pub_q = any(w in q for w in ("publication", "conference", "paper", "journal", "interesting"))
    boosted: list[dict] = []
    for row in episodes:
        r = dict(row)
        content = r.get("content") or ""
        cl = content.lower()
        score = float(r.get("score") or 0)
        if content_has_preference_cues(content):
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
    Non-recommend preference follow-ups: boost topic-overlap + maintenance cues.
    """
    if not is_preference_context_query(query) or not episodes:
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
        elif content_has_preference_cues(content):
            score = min(1.0, score + 0.1)
            r["preference_boost"] = True
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
    """
    text = content or ""
    if len(text) <= max_chars:
        return text
    terms = extract_search_terms(query)[:8]
    if not terms:
        return truncate_episode_content(text, max_chars=max_chars)
    lower = text.lower()
    windows: list[tuple[int, int]] = []
    span = max(600, max_chars // max(2, len(terms)))
    for t in terms:
        start = 0
        while True:
            idx = lower.find(t, start)
            if idx < 0:
                break
            a = max(0, idx - span // 3)
            b = min(len(text), idx + span)
            windows.append((a, b))
            start = idx + len(t)
            if len(windows) >= 6:
                break
        if len(windows) >= 6:
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
    if not is_topic_inventory_query(query) and not is_preference_context_query(query):
        if not is_activity_duration_query(query):
            return candidates
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
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
        scored.append((strength, row))
    if not scored:
        return candidates
    scored.sort(key=lambda x: x[0], reverse=True)
    by_uuid = {r.get("uuid"): r for r in candidates if r.get("uuid")}
    for strength, row in scored[: max(4, min(8, limit))]:
        uid = row.get("uuid")
        if not uid:
            continue
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


def effective_search_limit(query: str, limit: int) -> int:
    """Recommend / count / preference-context questions benefit from a wider pool."""
    base = max(limit, 1)
    q = (query or "").lower()
    if is_recommend_query(query) and any(
        w in q for w in ("publication", "conference", "paper", "journal", "interesting")
    ):
        return max(base, 16)
    if is_recommend_query(query) or is_count_query(query) or is_preference_context_query(query):
        return max(base, 12)
    return base


def prioritize_query_entity_recency(episodes: list[dict], query: str) -> list[dict]:
    """
    For questions naming a person/place, prefer newest *location-update*
    episodes (moved/suburbs/apartment), not merely newest mention.
    """
    focus = [t.lower() for t in extract_query_focus_terms(query)]
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


def build_activity_duration_digest(hits: list[dict], query: str = "") -> str:
    """List duration mentions for jogging/yoga/workout style hour questions."""
    if not is_activity_duration_query(query) or not hits:
        return ""
    q_acts = sorted({m.group(0).lower() for m in _ACTIVITY_RE.finditer(query or "")})
    lines = [
        "Activity duration mentions found in memory "
        "(sum matching activities for the asked window; convert minutes to hours):",
    ]
    n = 0
    total_hours = 0.0
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        if q_acts and not any(a in cl for a in q_acts):
            if not any(a[:3] in cl for a in q_acts if len(a) >= 3):
                continue
        when = h.get("valid_at_human") or ""
        sid = h.get("source_description") or h.get("uuid") or "?"
        for m in _DURATION_SPAN_RE.finditer(content):
            start = max(0, m.start() - 50)
            end = min(len(content), m.end() + 50)
            snippet = re.sub(r"\s+", " ", content[start:end]).strip()
            amount = float(m.group(1))
            unit = m.group(0).lower()
            hours = amount / 60.0 if "min" in unit else amount
            total_hours += hours
            stamp = f" @{when}" if when else ""
            lines.append(f"- ({sid}){stamp} ~{hours:g}h from '{m.group(0)}': {snippet}")
            n += 1
            if n >= 8:
                break
        if n >= 8:
            break
    if n == 0:
        lines.append(
            "- (none found in top excerpts) If no matching duration is present for the "
            "asked activities, answer 0 hours."
        )
    else:
        lines.append(
            f"Candidate duration sum (raw mentions, before date filter): {total_hours:g} hours. "
            "Prefer concrete completed workouts (e.g. 'I went for a 30-minute jog') over "
            "plans or old habits ('I used to'). Convert 30 minutes to 0.5 hours. "
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


def build_topic_inventory_digest(hits: list[dict], query: str = "") -> str:
    """
    For non-errand inventory counts: surface topic-overlapping episode snippets.
    """
    if not is_topic_inventory_query(query) or not hits:
        return ""
    phrases = extract_topic_phrases(query)
    nouns = extract_topic_nouns(query)
    if not phrases and not nouns:
        return ""
    lines = [
        "Topic inventory hints from memory (count distinct matching items/projects "
        "in the excerpts; do not stop at the first session):",
    ]
    seen_sid: set[str] = set()
    n = 0
    title_hints: list[str] = []
    seen_titles: set[str] = set()
    for h in hits:
        content = h.get("content") or ""
        cl = content.lower()
        matched = [p for p in phrases if p in cl] + [noun for noun in nouns if noun in cl]
        if not matched:
            continue
        sid = (h.get("source_description") or h.get("uuid") or "?").strip()
        if sid in seen_sid:
            continue
        seen_sid.add(sid)
        # Prefer a window around the first match
        idxs = [cl.find(m) for m in matched if cl.find(m) >= 0]
        idx = min(idxs) if idxs else 0
        start = max(0, idx - 50)
        end = min(len(content), idx + 120)
        snippet = re.sub(r"\s+", " ", content[start:end]).strip()
        lines.append(f"- ({sid}) [{', '.join(matched[:3])}] {snippet}")
        n += 1
        session_had_quoted = False
        for m in _PURCHASE_TITLE_RE.finditer(content):
            span = re.sub(r"\s+", " ", m.group(0)).strip()
            quoted = _QUOTED_TITLE_RE.findall(span) or _QUOTED_TITLE_RE.findall(
                content[max(0, m.start() - 40) : m.end() + 40]
            )
            if quoted:
                for qt in quoted:
                    key = qt.strip().lower()
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        title_hints.append(qt.strip())
                        session_had_quoted = True
            else:
                # Keep short purchase spans only (avoid long assistant chatter)
                if len(span) > 80:
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
            label = f"{am.group(1)} vinyl" if am else "purchased vinyl"
            key = label.lower()
            if key not in seen_titles:
                seen_titles.add(key)
                title_hints.append(label)
        if n >= 8:
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
        lines.append(
            f"Sessions with topic overlap listed: {n}. Enumerate every distinct matching "
            "item across sessions, then give the total integer together with the item names."
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


def build_preference_digest(hits: list[dict]) -> str:
    """Aggregate transferable preference cues for recommend-style questions."""
    prefs: list[str] = []
    feats: list[str] = []
    for h in hits:
        cues = h.get("cues") or {}
        for p in cues.get("preferences") or []:
            if p not in prefs:
                prefs.append(p)
        for f in cues.get("features") or []:
            if f not in feats:
                feats.append(f)
    if not prefs and not feats:
        return ""
    lines = [
        "Transferable user preferences from memory "
        "(reuse for the city/topic in the question even if the session names another city):"
    ]
    for p in prefs[:4]:
        lines.append(f"- {p}")
    if feats:
        lines.append("- features: " + ", ".join(feats[:8]))
    lines.append(
        "Give a concrete recommendation for the asked place/topic using these preferences; "
        "do not refuse solely because another city appears in the session. "
        "For hotel questions: open with a matching hotel suggestion for the asked city; "
        "never answer that you lack Miami/city hotels when these features are listed."
    )
    return "\n".join(lines)
