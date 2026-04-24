from __future__ import annotations

from dataclasses import dataclass

from lab17.memory import MemoryRecord, estimate_tokens


@dataclass(frozen=True, slots=True)
class ContextItem:
    content: str
    priority: int
    source: str
    token_count: int


@dataclass(frozen=True, slots=True)
class ContextBuildResult:
    items: list[ContextItem]
    used_tokens: int
    budget_tokens: int
    evicted: list[ContextItem]


class ContextWindowManager:
    """Auto-trims context using a four-level priority hierarchy.

    Priority 1: durable preferences and facts.
    Priority 2: episodic or semantic memories directly relevant to the query.
    Priority 3: recent conversation snippets.
    Priority 4: low-signal filler and older turns.
    """

    def __init__(self, max_tokens: int = 220, trim_ratio: float = 0.85) -> None:
        self.max_tokens = max_tokens
        self.trim_ratio = trim_ratio

    @property
    def budget_tokens(self) -> int:
        return int(self.max_tokens * self.trim_ratio)

    def build(self, query: str, records: list[MemoryRecord]) -> ContextBuildResult:
        items = [
            ContextItem(
                content=record.content,
                priority=min(max(record.priority, 1), 4),
                source=record.memory_type,
                token_count=record.token_count,
            )
            for record in records
        ]
        query_item = ContextItem(
            content=f"Current query: {query}",
            priority=1,
            source="current_query",
            token_count=estimate_tokens(query),
        )
        candidates = [query_item, *items]
        kept: list[ContextItem] = []
        evicted: list[ContextItem] = []

        for item in sorted(candidates, key=lambda value: (value.priority, value.token_count)):
            if sum(existing.token_count for existing in kept) + item.token_count <= self.budget_tokens:
                kept.append(item)
            else:
                evicted.append(item)

        kept.sort(key=lambda value: (value.priority, value.source))
        used_tokens = sum(item.token_count for item in kept)
        return ContextBuildResult(
            items=kept,
            used_tokens=used_tokens,
            budget_tokens=self.budget_tokens,
            evicted=evicted,
        )
