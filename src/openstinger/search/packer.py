"""Token-budget packing for retrieval responses."""

from __future__ import annotations

from typing import Any


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def pack_items(
    items: list[dict],
    *,
    max_tokens: int | None,
    text_key: str = "content",
) -> tuple[list[dict], int, int]:
    """
    Pack ranked items until token budget is exhausted.

    Truncates at item boundaries only. Always includes the top item
    (flagged truncated=true if over budget). Returns (packed, tokens_used, dropped).
    """
    if not items:
        return [], 0, 0
    if max_tokens is None or max_tokens <= 0:
        used = sum(_count_tokens(str(it.get(text_key) or it.get("fact") or it.get("text") or "")) for it in items)
        return items, used, 0

    packed: list[dict] = []
    used = 0
    for i, it in enumerate(items):
        text = str(it.get(text_key) or it.get("fact") or it.get("text") or "")
        n = _count_tokens(text)
        if i == 0:
            row = dict(it)
            if n > max_tokens:
                # Keep head of text within budget
                approx_chars = max_tokens * 4
                row[text_key] = text[:approx_chars] + ("…" if len(text) > approx_chars else "")
                if text_key == "content":
                    row["content"] = row[text_key]
                row["truncated"] = True
                n = _count_tokens(row.get(text_key) or "")
            packed.append(row)
            used += n
            continue
        if used + n > max_tokens:
            break
        packed.append(it)
        used += n
    dropped = max(0, len(items) - len(packed))
    return packed, used, dropped
