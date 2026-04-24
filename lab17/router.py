from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteDecision:
    intent: str
    backends: list[str]
    priority: int
    reason: str


class MemoryRouter:
    """Routes user turns to the most useful memory backends."""

    preference_markers = {
        "like",
        "prefer",
        "favorite",
        "thich",
        "muon",
        "hay",
        "always",
        "never",
    }
    factual_markers = {
        "deadline",
        "email",
        "id",
        "name",
        "when",
        "where",
        "who",
        "which",
    }
    experience_markers = {
        "remember",
        "last time",
        "last paper",
        "yesterday",
        "visited",
        "meeting",
        "discussion",
        "trip",
        "story",
    }

    def route(self, query: str) -> RouteDecision:
        normalized = query.lower()
        words = set(normalized.replace("?", " ").replace(",", " ").split())

        if any(marker in normalized for marker in self.experience_markers):
            return RouteDecision(
                intent="experience_recall",
                backends=["json_episodic_log", "chroma_semantic", "conversation_buffer"],
                priority=2,
                reason="Query asks for past events or experiences.",
            )

        if words & self.preference_markers:
            return RouteDecision(
                intent="user_preference",
                backends=["redis_long_term", "chroma_semantic", "conversation_buffer"],
                priority=1,
                reason="Query contains preference markers.",
            )

        if words & self.factual_markers:
            return RouteDecision(
                intent="factual_recall",
                backends=["redis_long_term", "chroma_semantic", "conversation_buffer"],
                priority=1,
                reason="Query asks for a fact-like answer.",
            )

        if normalized.endswith("?"):
            return RouteDecision(
                intent="factual_recall",
                backends=["redis_long_term", "chroma_semantic", "conversation_buffer"],
                priority=1,
                reason="Question defaults to factual recall.",
            )

        return RouteDecision(
            intent="general_context",
            backends=["conversation_buffer", "chroma_semantic"],
            priority=3,
            reason="Default to recent context and semantic recall.",
        )
