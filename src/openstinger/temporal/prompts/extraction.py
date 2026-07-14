"""LLM prompts for entity and edge extraction."""

from __future__ import annotations

EXTRACT_ENTITIES_SYSTEM = """You are an entity extraction assistant.
Extract all named entities (people, organisations, concepts, locations, events) from the provided text.
For each entity provide:
- name: the canonical name as it appears (e.g. "Alice Smith", "Acme Corp")
- entity_type: one of PERSON, ORG, CONCEPT, LOCATION, EVENT, ENTITY
- summary: one sentence describing the entity based on context

Be conservative: only extract entities clearly mentioned. Do not infer entities not present."""


EXTRACT_EDGES_SYSTEM = """You are a relationship extraction assistant.
Given a text and a list of known entities, extract factual relationships between pairs of entities.
For each relationship provide:
- source_entity_name: name of the source entity (must be from the provided list)
- target_entity_name: name of the target entity (must be from the provided list)
- relation_type: a short uppercase label (e.g. WORKS_AT, KNOWS, LOCATED_IN, OWNS, PART_OF)
- fact: a concise factual statement (e.g. "Alice Smith works at Acme Corp as an engineer")
- valid_from_iso: ISO 8601 date if the text specifies when this became true, otherwise null
- valid_to_iso: ISO 8601 date if the text specifies when this stopped being true, otherwise null

Resolve relative dates against the episode reference time provided in the user message
(e.g. if reference_time is 2026-03-10 and text says "last Friday", valid_from_iso is 2026-03-06).
Only extract relationships that are explicitly stated or strongly implied by the text.
Do not create relationships between entities that are merely mentioned in the same sentence."""


EXTRACT_STATEMENTS_SYSTEM = """You are a memory distillation assistant.
Decompose the episode into at most 12 atomic, self-contained statements.
Rules:
- Resolve pronouns to explicit subjects (e.g. "User prefers…", "Rachel moved…")
- State preferences, actions, errands, and expertise as declarative facts
- Include dates when stated; resolve relative dates against the provided reference_time
- Do not invent facts not present in the text
For each statement provide:
- text: one sentence, subject explicit
- kind: one of fact | preference | expertise | errand | location_update
- valid_from_iso: ISO date if known or resolved, else null

Kind guidance (generic product memory, not domain-specific tags):
- preference: likes, wants, hotel features, recommendation criteria
- expertise: research focus, professional specialty, demonstrated skill depth
- errand: pickup, return, exchange, redeem, buy, or similar pending/completed tasks
- location_update: moved, relocated, current city/home/place for a person
- fact: everything else that is still worth remembering
"""


def build_extract_entities_user(content: str) -> str:
    return f"Extract all named entities from this text:\n\n{content}"


def build_extract_edges_user(
    content: str,
    entity_names: list[str],
    reference_time: str | None = None,
) -> str:
    names_str = ", ".join(f'"{n}"' for n in entity_names)
    ref = f"\nEpisode reference_time: {reference_time}\n" if reference_time else "\n"
    return (
        f"Known entities: [{names_str}]{ref}\n"
        f"Extract factual relationships between these entities from:\n\n{content}"
    )


def build_extract_statements_user(content: str, reference_time: str | None = None) -> str:
    ref = f"Episode reference_time: {reference_time}\n\n" if reference_time else ""
    return f"{ref}Distill atomic statements from:\n\n{content}"
