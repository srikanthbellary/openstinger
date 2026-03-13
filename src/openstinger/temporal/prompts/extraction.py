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

Only extract relationships that are explicitly stated or strongly implied by the text.
Do not create relationships between entities that are merely mentioned in the same sentence."""


def build_extract_entities_user(content: str) -> str:
    return f"Extract all named entities from this text:\n\n{content}"


def build_extract_edges_user(content: str, entity_names: list[str]) -> str:
    names_str = ", ".join(f'"{n}"' for n in entity_names)
    return (
        f"Known entities: [{names_str}]\n\n"
        f"Extract factual relationships between these entities from:\n\n{content}"
    )
