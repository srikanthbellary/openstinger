"""
Quarantined LME-informed lexical expanders and domain heuristics.

Default-off behind ``retrieval.experimental_lexicons``. Do not import these
into default-on retrieval paths when publishing fair scores.
"""

from __future__ import annotations

import re

KIT_EXPAND = (
    "model kit", "scale model", "plastic model", "revell", "tamiya",
    "spitfire", "eagle",
)
PUB_EXPAND = (
    "publication", "conference", "journal", "workshop", "paper",
    "proceedings", "symposium",
)
ERRAND_EXPAND = (
    "dry cleaning", "dry cleaners", "pick up", "return", "exchange",
    "clothing", "boots", "blazer",
)
DOMAIN_LEXICONS: dict[str, tuple[str, ...]] = {
    "healthcare_imaging": (
        "medical image", "medical imaging", "radiology", "pathology",
        "healthcare", "clinical", "miccai", "brats", "segmentation",
        "mri", "ct scan", "x-ray", "diagnostic imaging",
    ),
    "robotics": (
        "robotics", "autonomous", "slam", "manipulation", "humanoid",
    ),
    "climate_env": (
        "climate", "sustainability", "environmental", "carbon", "ecology",
    ),
    "nlp_llm": (
        "natural language", "language model", "nlp", "transformer", "llm",
    ),
}
STORE_NAMES = {
    "target", "walmart", "costco", "amazon", "best buy", "home depot",
    "zara", "ikea", "cvs", "walgreens", "kroger", "trader joe", "whole foods",
    "apple store", "nordstrom", "macy", "sephora", "starbucks",
}
HOTEL_FEATURE_RE = re.compile(
    r"(?:rooftop\s+pool|hot tub(?:\s+on the balcony)?|ocean view|city skyline|"
    r"great views?|private balcony|floor-to-ceiling)",
    re.I,
)
SOP_NOISE_RE = re.compile(
    r"\b(?:statement of purpose|\bsop\b|admissions(?:\s+committee)?|"
    r"graduate program|masters? application|phd application|"
    r"why (?:this|your) (?:program|university|school))\b",
    re.I,
)
HOTEL_FOCUS_EXTRAS = ("hotel", "view", "rooftop", "balcony", "prefer")
PUB_FOCUS_EXTRAS = (
    "my research",
    "working in the field",
    "research interest",
    "skip the basics",
    "deep learning for",
)


def append_experimental_subqueries(query: str, add_fn) -> None:
    """Call ``add_fn(phrase)`` for each experimental expand phrase."""
    ql = (query or "").lower()
    if "kit" in ql or "model" in ql:
        for s in KIT_EXPAND:
            add_fn(s)
    if any(w in ql for w in ("publication", "conference", "paper", "journal", "interesting")):
        for s in PUB_EXPAND:
            add_fn(s)
    from openstinger.temporal.search_utils import is_count_query
    if is_count_query(query) or any(
        w in ql for w in ("pick up", "return", "clothing", "store", "exchange")
    ):
        for s in ERRAND_EXPAND:
            add_fn(s)
