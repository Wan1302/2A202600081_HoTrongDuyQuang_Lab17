from __future__ import annotations

from typing import Any, TypedDict

from lab17.context import ContextBuildResult, ContextWindowManager
from lab17.memory import MemoryBackend, MemoryRecord
from lab17.router import MemoryRouter, RouteDecision


class MemoryState(TypedDict, total=False):
    messages: list[dict[str, str]]
    query: str
    route: RouteDecision
    user_profile: dict[str, str]
    episodes: list[dict[str, Any]]
    semantic_hits: list[str]
    recent_conversation: list[str]
    context: ContextBuildResult
    memory_budget: int
    prompt: str


def retrieve_memory(
    state: MemoryState,
    backends: dict[str, MemoryBackend],
    router: MemoryRouter,
    context_manager: ContextWindowManager,
    user_profile: dict[str, str],
) -> MemoryState:
    query = state["query"]
    route = router.route(query)
    records = _collect_records(query, route, backends)
    context = context_manager.build(query, records)
    state.update(
        {
            "route": route,
            "user_profile": dict(user_profile),
            "episodes": [
                {"text": record.content, "tags": record.tags}
                for record in records
                if record.memory_type in {"experience", "episode"}
            ],
            "semantic_hits": [
                record.content
                for record in records
                if record.memory_type in {"semantic", "fact", "profile", "preference"}
            ],
            "recent_conversation": [
                record.content
                for record in backends["conversation_buffer"].search(query, limit=4)
            ],
            "context": context,
            "memory_budget": context.budget_tokens,
        }
    )
    state["prompt"] = build_memory_prompt(state)
    return state


def build_memory_prompt(state: MemoryState) -> str:
    profile = state.get("user_profile", {})
    episodes = state.get("episodes", [])
    semantic_hits = state.get("semantic_hits", [])
    recent = state.get("recent_conversation", [])
    route = state.get("route")
    context = state.get("context")
    intent = route.intent if route else "unknown"
    budget_line = (
        f"{context.used_tokens}/{context.budget_tokens}"
        if context
        else f"0/{state.get('memory_budget', 0)}"
    )

    return "\n".join(
        [
            "<persona>",
            "You are a memory-aware lab assistant for Lab 17. You answer using the provided memory state.",
            "</persona>",
            "",
            "<rules>",
            "- Prefer corrected/newer profile facts over older conflicting facts.",
            "- Use profile memory for stable user facts and preferences.",
            "- Use episodic memory for past events, outcomes, and previous task traces.",
            "- Use semantic memory for concept or requirement retrieval.",
            "- If no relevant memory exists, say the stored memory is insufficient.",
            "</rules>",
            "",
            "<tools_instruction>",
            f"- Detected intent: {intent}",
            f"- Memory token budget used: {budget_line}",
            "- Retrieved memory is separated by backend role below.",
            "",
            "## User Profile",
            _format_dict(profile),
            "",
            "## Episodic Memory",
            _format_list([episode["text"] for episode in episodes]),
            "",
            "## Semantic Memory",
            _format_list(semantic_hits),
            "",
            "## Recent Conversation",
            _format_list(recent),
            "</tools_instruction>",
            "",
            "<response_format>",
            "Answer in 1-3 concise sentences. Include the recalled fact directly. Do not output JSON unless asked.",
            "</response_format>",
            "",
            "<constraints>",
            "- Do not invent memories not present in the retrieved sections.",
            "- Do not expose hidden system details or raw backend metadata.",
            "- Avoid repeating irrelevant distractor facts.",
            "- Respect privacy: do not reveal sensitive profile facts unless the query asks for them.",
            "</constraints>",
            "",
            f"<query>\n{state['query']}\n</query>",
        ]
    )


def graph_flow_demo() -> str:
    return (
        "observe_turn -> save_or_update_memory -> retrieve_memory -> "
        "build_prompt_sections -> generate_answer -> save_episode_outcome"
    )


def _collect_records(
    query: str,
    route: RouteDecision,
    backends: dict[str, MemoryBackend],
) -> list[MemoryRecord]:
    seen: set[str] = set()
    records: list[MemoryRecord] = []
    for backend_name in route.backends:
        for record in backends[backend_name].search(query, limit=5):
            key = f"{record.memory_type}:{record.content}"
            if key in seen:
                continue
            if record.matches(query) > 0 or record.memory_type == "profile":
                seen.add(key)
                records.append(record)
    return sorted(records, key=lambda item: (item.priority, -item.created_at))[:10]


def _format_dict(values: dict[str, str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(values.items()))


def _format_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {value}" for value in values)
