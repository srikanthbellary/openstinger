"""Prompt for memory_reflect QA over retrieved memories."""

from __future__ import annotations

REFLECT_SYSTEM = """You answer questions using only the provided memory excerpts.
Rules:
- Prefer NEWER / LATEST labeled facts when they conflict with older ones, unless the question asks about the past.
- For counting questions, enumerate matching items then give the integer total.
- For recommendations, scope to the user's demonstrated preferences or specialty in the memories.
- Perform date arithmetic carefully when asked about durations or order.
- If the memories are insufficient, reply exactly: INSUFFICIENT_MEMORY
- Be concise and factual. Cite supporting item indices like [1], [2].
- Do not include chain-of-thought or <think> tags.
"""


def build_reflect_user(
    query: str,
    blocks: list[str],
    conflicts: list[dict] | None = None,
) -> str:
    body = "\n\n".join(blocks) if blocks else "(no memories)"
    conflict_txt = ""
    if conflicts:
        lines = ["Conflict hints (prefer NEWER unless question is historical):"]
        for c in conflicts[:5]:
            lines.append(
                f"- {c.get('entity_hint')}: older@{c.get('older_valid_at_human') or c.get('older_valid_at')} "
                f"-> newer@{c.get('newer_valid_at_human') or c.get('newer_valid_at')}"
            )
        conflict_txt = "\n\n" + "\n".join(lines)
    return f"Question: {query}\n\nMemories:\n{body}{conflict_txt}\n\nAnswer:"
