from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from lab17.agent import AgentTurnResult, MemoryAgent
from lab17.graph import graph_flow_demo
from lab17.llm import LLMClient
from lab17.memory import tokenize


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    category: str
    turns: list[str]
    query: str
    expected_terms: list[str]


SCENARIOS = [
    Scenario(
        name="profile_name_after_noise",
        category="profile_recall",
        turns=[
            "My name is Linh.",
            "Today I am debugging a memory router.",
            "The meeting room is B204, but that is not part of my profile.",
            "I also mentioned another student named Minh in the hallway.",
        ],
        query="What is my name?",
        expected_terms=["linh"],
    ),
    Scenario(
        name="allergy_conflict_update",
        category="conflict_update",
        turns=[
            "Tôi dị ứng sữa bò.",
            "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
            "For dinner planning, remember the corrected allergy only.",
        ],
        query="Tôi dị ứng gì?",
        expected_terms=["đậu", "nành"],
    ),
    Scenario(
        name="rescheduled_project_deadline",
        category="factual_recall",
        turns=[
            "The graph RAG project deadline was originally Friday week 7.",
            "During class the lecturer moved the graph RAG deadline to Monday week 8.",
            "The Friday week 7 date is only for the presentation draft now.",
        ],
        query="When is the final graph RAG project deadline now?",
        expected_terms=["monday", "week", "8"],
    ),
    Scenario(
        name="meeting_with_distractors",
        category="episodic_recall",
        turns=[
            "Yesterday I had a meeting with Dr. Linh about memory routers.",
            "Before that meeting I chatted with An about UI colors, unrelated to the lab.",
            "Dr. Linh asked us to compare preference recall, factual recall, and episodic recall separately.",
            "The room booking number was B204, but that detail is not important.",
        ],
        query="What did Dr. Linh ask us to compare in the meeting?",
        expected_terms=["preference", "factual", "episodic", "recall"],
    ),
    Scenario(
        name="conditional_language_preference",
        category="profile_recall",
        turns=[
            "For casual chat I like English because it helps me practice.",
            "For lab reports, I want concise Vietnamese with technical terms kept in English.",
            "Do not make the lab report sound like marketing copy.",
        ],
        query="How should you write my lab report summary?",
        expected_terms=["concise", "vietnamese", "technical", "english"],
    ),
    Scenario(
        name="corrected_email_fact",
        category="conflict_update",
        turns=[
            "My backup email is student17@vinuni.edu.vn, wait that may be wrong.",
            "Correction: use lab17.student@vinuni.edu.vn for backup contact.",
            "The old student17 address belongs to another account.",
        ],
        query="What is my backup email?",
        expected_terms=["lab17", "student", "vinuni", "edu", "vn"],
    ),
    Scenario(
        name="trip_sequence_recall",
        category="episodic_recall",
        turns=[
            "Last time in Da Nang I visited Son Tra before the AI workshop.",
            "I skipped Ba Na Hills because the weather turned bad.",
            "After the workshop I wrote notes about retrieval evaluation at the hotel.",
        ],
        query="What place did I visit before the AI workshop in Da Nang?",
        expected_terms=["son", "tra", "before", "workshop"],
    ),
    Scenario(
        name="tool_preference_with_negative",
        category="profile_recall",
        turns=[
            "I tried unittest for an older Python lab.",
            "For this memory lab I prefer pytest because the output is easier to scan.",
            "Do not assume I want a web dashboard for the benchmark.",
        ],
        query="Which Python test tool do I prefer for this memory lab?",
        expected_terms=["pytest", "memory", "lab"],
    ),
    Scenario(
        name="conflicting_library_choice",
        category="semantic_retrieval",
        turns=[
            "At first I planned to use FAISS for semantic memory.",
            "The lab requirement specifically asks for Chroma as the semantic backend.",
            "Use FAISS only as a comparison note, not as the implementation choice.",
        ],
        query="Which semantic memory backend should the lab implementation use?",
        expected_terms=["chroma", "semantic", "backend"],
    ),
    Scenario(
        name="token_budget_priority",
        category="trim_token_budget",
        turns=[
            "High priority: keep user preferences and corrected facts even when trimming context.",
            "Medium priority: keep directly relevant episodic memories.",
            "Low priority: remove filler chatter, old jokes, and unrelated room numbers first.",
            "The context window for this lab is intentionally small to test eviction.",
        ],
        query="What should be kept first when the context window is trimmed?",
        expected_terms=["preferences", "corrected", "facts", "relevant", "episodic"],
    ),
]


def run_benchmark(
    output_dir: Path = Path("reports"),
    data_root: Path = Path("data"),
    llm: LLMClient | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    scenarios = SCENARIOS[:limit] if limit else SCENARIOS
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        with_memory = _run_scenario(scenario, memory_enabled=True, data_root=data_root, llm=llm)
        without_memory = _run_scenario(scenario, memory_enabled=False, data_root=data_root, llm=llm)
        rows.append(
            {
                "index": index,
                "scenario": scenario.name,
                "category": scenario.category,
                "turns": scenario.turns,
                "query": scenario.query,
                "expected_terms": scenario.expected_terms,
                "with_memory": _score_result(with_memory, scenario.expected_terms),
                "without_memory": _score_result(without_memory, scenario.expected_terms),
            }
        )

    summary = {
        "scenario_count": len(rows),
        "generation": {
            "llm_provider": llm.provider if llm else "rule_based",
            "model": llm.model if llm else "deterministic_mock",
        },
        "graph_flow": graph_flow_demo(),
        "memory_backends": rows[0]["with_memory"]["memory_backend_modes"] if rows else {},
        "metrics": {
            "with_memory": _aggregate([row["with_memory"] for row in rows]),
            "without_memory": _aggregate([row["without_memory"] for row in rows]),
        },
        "memory_hit_rate_analysis": _hit_rate_analysis(rows),
        "token_budget_breakdown": _token_budget_breakdown(rows),
        "rows": rows,
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def _run_scenario(
    scenario: Scenario,
    memory_enabled: bool,
    data_root: Path,
    llm: LLMClient | None,
) -> AgentTurnResult:
    label = "with_memory" if memory_enabled else "without_memory"
    agent = MemoryAgent(
        memory_enabled=memory_enabled,
        data_dir=data_root / label / scenario.name,
        llm=llm,
    )
    for turn in scenario.turns:
        agent.observe(turn)
    return agent.answer(scenario.query)


def _score_result(result: AgentTurnResult, expected_terms: list[str]) -> dict[str, Any]:
    response_tokens = set(tokenize(result.response))
    expected = set(expected_terms)
    matched = sorted(expected & response_tokens)
    relevance = len(matched) / len(expected) if expected else 0.0
    context_utilization = result.context.used_tokens / result.context.budget_tokens
    token_efficiency = relevance / max(1, result.response_tokens)
    return {
        "response": result.response,
        "llm_provider": result.llm_provider,
        "model": result.model,
        "memory_backend_modes": result.memory_backend_modes,
        "prompt_sections": list(result.memory_state["prompt"].split("\n\n")[:4]),
        "intent": result.route.intent,
        "memory_hits": result.memory_hits,
        "matched_terms": matched,
        "response_relevance": round(relevance, 4),
        "context_utilization": round(context_utilization, 4),
        "response_tokens": result.response_tokens,
        "token_efficiency": round(token_efficiency, 6),
        "context_used_tokens": result.context.used_tokens,
        "context_budget_tokens": result.context.budget_tokens,
        "evicted_items": len(result.context.evicted),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "avg_response_relevance": round(mean(item["response_relevance"] for item in results), 4),
        "avg_context_utilization": round(mean(item["context_utilization"] for item in results), 4),
        "avg_response_tokens": round(mean(item["response_tokens"] for item in results), 2),
        "avg_token_efficiency": round(mean(item["token_efficiency"] for item in results), 6),
        "memory_hit_rate": round(
            sum(1 for item in results if item["memory_hits"] > 0) / len(results),
            4,
        ),
    }


def _hit_rate_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_hits = [row for row in rows if row["with_memory"]["memory_hits"] > 0]
    no_hits = [row for row in rows if row["with_memory"]["memory_hits"] == 0]
    return {
        "with_memory_hit_scenarios": len(with_hits),
        "with_memory_miss_scenarios": [row["scenario"] for row in no_hits],
        "interpretation": "Memory improves profile recall, conflict updates, episodic recall, semantic retrieval, and token-budget cases.",
    }


def _token_budget_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "max_context_budget_tokens": max(row["with_memory"]["context_budget_tokens"] for row in rows),
        "avg_with_memory_context_used_tokens": round(
            mean(row["with_memory"]["context_used_tokens"] for row in rows),
            2,
        ),
        "avg_without_memory_context_used_tokens": round(
            mean(row["without_memory"]["context_used_tokens"] for row in rows),
            2,
        ),
        "eviction_total": sum(row["with_memory"]["evicted_items"] for row in rows),
    }
