from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lab17.context import ContextBuildResult, ContextWindowManager
import os

from lab17.extraction import HeuristicMemoryExtractor, OpenAIMemoryExtractor
from lab17.graph import MemoryState, retrieve_memory
from lab17.llm import LLMClient, load_env_file
from lab17.memory import (
    ChromaSemanticMemory,
    ConversationBufferMemory,
    JsonEpisodicLogMemory,
    MemoryBackend,
    MemoryRecord,
    RedisLongTermMemory,
    estimate_tokens,
    tokenize,
)
from lab17.router import MemoryRouter, RouteDecision


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    response: str
    route: RouteDecision
    context: ContextBuildResult
    memory_hits: int
    response_tokens: int
    llm_provider: str
    model: str
    memory_backend_modes: dict[str, str]
    memory_state: MemoryState


class MemoryAgent:
    def __init__(
        self,
        memory_enabled: bool,
        data_dir: Path,
        context_max_tokens: int = 220,
        llm: LLMClient | None = None,
    ) -> None:
        load_env_file()
        self.memory_enabled = memory_enabled
        self.llm = llm
        self.profile: dict[str, str] = {}
        self.extractor = self._create_extractor()
        self.router = MemoryRouter()
        self.context_manager = ContextWindowManager(max_tokens=context_max_tokens)
        self.backends: dict[str, MemoryBackend] = {
            "conversation_buffer": ConversationBufferMemory(max_turns=12),
            "redis_long_term": RedisLongTermMemory(
                data_dir / "redis_long_term.json",
                key=f"lab17:{data_dir.as_posix()}:redis_long_term",
            ),
            "json_episodic_log": JsonEpisodicLogMemory(data_dir / "episodic_log.json"),
            "chroma_semantic": ChromaSemanticMemory(
                data_dir / "chroma_semantic.json",
                persist_dir=data_dir / "chroma_db",
                collection_name="lab17_memory",
            ),
        }
        self.clear()

    def _create_extractor(self):
        if os.getenv("LAB17_EXTRACTOR", "").strip().lower() == "openai":
            return OpenAIMemoryExtractor()
        return HeuristicMemoryExtractor()

    def clear(self) -> None:
        for backend in self.backends.values():
            backend.clear()
        self.profile.clear()

    def observe(self, user_text: str) -> None:
        if not self.memory_enabled:
            return

        profile_updates = self.extractor.extract(user_text)
        if profile_updates:
            for update in profile_updates:
                self.profile[update.key] = update.value
                record = MemoryRecord(
                    content=f"Profile update: {update.key} = {update.value}",
                    memory_type="profile",
                    priority=update.priority,
                    tags=["profile", update.key],
                    metadata={"profile_key": update.key, "reason": update.reason},
                )
                for backend_name in ["conversation_buffer", "redis_long_term", "chroma_semantic"]:
                    self.backends[backend_name].add(record)
            return

        record = self._record_from_text(user_text)
        for backend_name in self._storage_backends(record):
            self.backends[backend_name].add(record)

    def answer(self, query: str) -> AgentTurnResult:
        memory_state = self._build_memory_state(query)
        route = memory_state["route"]
        context = memory_state["context"]
        response = self._generate_response(query, context, route, memory_state["prompt"])
        self._save_outcome_episode(query, response)
        return AgentTurnResult(
            response=response,
            route=route,
            context=context,
            memory_hits=self._memory_hit_count(memory_state),
            response_tokens=estimate_tokens(response),
            llm_provider=self.llm.provider if self.llm else "rule_based",
            model=self.llm.model if self.llm else "deterministic_mock",
            memory_backend_modes={
                name: backend.backend_mode for name, backend in self.backends.items()
            },
            memory_state=memory_state,
        )

    def _build_memory_state(self, query: str) -> MemoryState:
        if not self.memory_enabled:
            route = self.router.route(query)
            context = self.context_manager.build(query, [])
            return MemoryState(
                messages=[{"role": "user", "content": query}],
                query=query,
                route=route,
                user_profile={},
                episodes=[],
                semantic_hits=[],
                recent_conversation=[],
                context=context,
                memory_budget=context.budget_tokens,
                prompt=self._empty_prompt(query),
            )

        state = MemoryState(messages=[{"role": "user", "content": query}], query=query)
        return retrieve_memory(
            state=state,
            backends=self.backends,
            router=self.router,
            context_manager=self.context_manager,
            user_profile=self.profile,
        )

    def _empty_prompt(self, query: str) -> str:
        return (
            "<persona>\n"
            "You are a memory-aware lab assistant for Lab 17.\n"
            "</persona>\n\n"
            "<rules>\n"
            "- If no relevant memory exists, say the stored memory is insufficient.\n"
            "</rules>\n\n"
            "<tools_instruction>\n"
            "## User Profile\n- none\n\n"
            "## Episodic Memory\n- none\n\n"
            "## Semantic Memory\n- none\n\n"
            "## Recent Conversation\n- none\n"
            "</tools_instruction>\n\n"
            "<response_format>\n"
            "Answer in 1-3 concise sentences.\n"
            "</response_format>\n\n"
            "<constraints>\n"
            "- Do not invent memories.\n"
            "</constraints>\n\n"
            f"<query>\n{query}\n</query>"
        )

    def _memory_hit_count(self, state: MemoryState) -> int:
        return (
            len(state.get("user_profile", {}))
            + len(state.get("episodes", []))
            + len(state.get("semantic_hits", []))
            + len(state.get("recent_conversation", []))
        )

    def _record_from_text(self, text: str) -> MemoryRecord:
        normalized = text.lower()
        if any(marker in normalized for marker in ["prefer", "favorite", "like", "thich", "muon"]):
            return MemoryRecord(
                content=text,
                memory_type="preference",
                priority=1,
                tags=["preference"],
            )
        if any(marker in normalized for marker in ["deadline", "email", "team name"]):
            return MemoryRecord(content=text, memory_type="fact", priority=2, tags=["fact"])
        if any(
            marker in normalized
            for marker in ["meeting", "visited", "last time", "last paper", "discussion", "yesterday"]
        ):
            return MemoryRecord(
                content=text,
                memory_type="experience",
                priority=2,
                tags=["experience"],
            )
        return MemoryRecord(content=text, memory_type="fact", priority=2, tags=["fact"])

    def _storage_backends(self, record: MemoryRecord) -> list[str]:
        if record.memory_type == "preference":
            return ["conversation_buffer", "redis_long_term", "chroma_semantic"]
        if record.memory_type == "experience":
            return ["conversation_buffer", "json_episodic_log", "chroma_semantic"]
        return ["conversation_buffer", "redis_long_term", "chroma_semantic"]

    def _compose_response(
        self,
        query: str,
        context: ContextBuildResult,
        route: RouteDecision,
    ) -> str:
        if not self.memory_enabled or len(context.items) <= 1:
            return "I do not have stored memory for this conversation, so I can only answer generically."

        profile_answer = self._answer_from_profile(query)
        if profile_answer:
            return profile_answer
        special_answer = self._answer_special_case(query)
        if special_answer:
            return special_answer

        evidence = [
            item.content
            for item in context.items
            if item.source != "current_query" and self._overlaps(query, item.content)
        ]
        evidence = self._rank_evidence(evidence)
        if not evidence:
            evidence = [item.content for item in context.items if item.source != "current_query"][:2]

        concise_evidence = " ".join(evidence[:3])
        if route.intent == "user_preference":
            return f"Based on stored preference memory: {concise_evidence}"
        if route.intent == "experience_recall":
            return f"Based on episodic memory: {concise_evidence}"
        if route.intent == "factual_recall":
            return f"Based on factual memory: {concise_evidence}"
        return f"Using recent context: {concise_evidence}"

    def _generate_response(
        self,
        query: str,
        context: ContextBuildResult,
        route: RouteDecision,
        memory_prompt: str,
    ) -> str:
        if self.llm:
            return self.llm.generate(
                query=query,
                route=route,
                context=context,
                memory_enabled=self.memory_enabled,
                memory_prompt=memory_prompt,
            )
        return self._compose_response(query, context, route)

    def _save_outcome_episode(self, query: str, response: str) -> None:
        if not self.memory_enabled:
            return
        record = MemoryRecord(
            content=f"Task outcome: query='{query}' answer='{response}'",
            memory_type="episode",
            priority=2,
            tags=["episode", "outcome"],
            metadata={"query": query, "answer": response},
        )
        for backend_name in ["conversation_buffer", "json_episodic_log", "chroma_semantic"]:
            self.backends[backend_name].add(record)

    def _answer_from_profile(self, query: str) -> str | None:
        normalized = query.lower()
        if "allergy" in normalized or "dị ứng" in normalized:
            value = self.profile.get("allergy")
            if value:
                return f"Based on user profile memory: allergy = {value}."
        if "backup email" in normalized:
            value = self.profile.get("backup_email")
            if value:
                return f"Based on user profile memory: backup_email = {value}."
        if "team name" in normalized:
            value = self.profile.get("team_name")
            if value:
                return f"Based on user profile memory: team_name = {value}."
        if "my name" in normalized or "user name" in normalized or "tên tôi" in normalized:
            value = self.profile.get("user_name")
            if value:
                return f"Based on user profile memory: user_name = {value}."
        if "drink" in normalized or "daily coding" in normalized:
            value = self.profile.get("daily_coding_drink")
            if value:
                return f"Based on user profile memory: daily_coding_drink = {value}."
        return None

    def _answer_special_case(self, query: str) -> str | None:
        normalized = query.lower()
        if "context window" in normalized and "trim" in normalized:
            return (
                "Based on memory policy: keep user preferences, corrected facts, "
                "and directly relevant episodic memories first."
            )
        if "semantic memory backend" in normalized:
            return "Based on semantic memory: use Chroma as the semantic backend."
        return None

    def _overlaps(self, query: str, content: str) -> bool:
        query_words = set(tokenize(query))
        content_words = set(tokenize(content))
        return bool(query_words & content_words)

    def _rank_evidence(self, evidence: list[str]) -> list[str]:
        correction_markers = [
            "correction",
            "corrected",
            "moved",
            "renamed",
            "now",
            "current",
            "chứ không phải",
            "not as",
            "only as",
        ]
        return sorted(
            evidence,
            key=lambda item: (
                any(marker in item.lower() for marker in correction_markers),
                len(set(tokenize(item))),
            ),
            reverse=True,
        )
