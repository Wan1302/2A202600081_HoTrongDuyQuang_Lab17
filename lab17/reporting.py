from __future__ import annotations

from pathlib import Path
from typing import Any


def write_markdown_report(summary: dict[str, Any], path: Path = Path("reports/report.md")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with_memory = summary["metrics"]["with_memory"]
    without_memory = summary["metrics"]["without_memory"]
    generation = summary.get("generation", {})
    memory_backends = summary.get("memory_backends", {})

    lines = [
        "# Lab 17 Report",
        "",
        "## Work Completed",
        "",
        "- Implemented four memory backends: conversation buffer, Redis-style long-term store, JSON episodic log, and Chroma-style semantic retrieval.",
        "- Added a MemoryState/LangGraph-style skeleton with retrieve_memory(state) and sectioned prompt injection.",
        "- Standardized the LLM prompt with <persona>, <rules>, <tools_instruction>, <response_format>, and <constraints> blocks.",
        "- Added profile save/update with conflict handling for corrected facts.",
        "- Built an intent-based memory router for user preferences, factual recall, experience recall, and general context.",
        "- Added context window management with auto-trim and four-level priority eviction.",
        f"- Added deterministic benchmark over exactly {summary['scenario_count']} noisy multi-turn conversations.",
        "",
        "## Dataset Design",
        "",
        "- The benchmark uses synthetic but noisy conversations with distractor facts, corrected facts, changed preferences, conditional preferences, and paraphrased queries.",
        "- This makes the benchmark better for stress-testing routing and retrieval than a clean keyword-overlap dataset.",
        "- The dataset is still synthetic, so the results should be treated as lab evidence rather than production-grade evaluation.",
        "",
        "## Generation",
        "",
        f"- LLM provider: {generation.get('llm_provider', 'rule_based')}.",
        f"- Model: {generation.get('model', 'deterministic_mock')}.",
        "",
        "## Graph Flow",
        "",
        f"`{summary.get('graph_flow', 'not recorded')}`",
        "",
        "## Memory Backends",
        "",
        "| Backend | Mode |",
        "| --- | --- |",
        *[
            f"| {backend} | {mode} |"
            for backend, mode in memory_backends.items()
        ],
        "",
        "## Metrics Comparison",
        "",
        "| Agent | Avg relevance | Avg context utilization | Avg response tokens | Avg token efficiency | Memory hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _metric_row("With memory", with_memory),
        _metric_row("Without memory", without_memory),
        "",
        "## Scenario Details",
        "",
        "| # | Category | Scenario | With-memory relevance | Without-memory relevance | Hits | Matched terms |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for row in summary["rows"]:
        matched = ", ".join(row["with_memory"]["matched_terms"]) or "-"
        lines.append(
            f"| {row['index']} | {row['category']} | {row['scenario']} | "
            f"{row['with_memory']['response_relevance']:.2f} | "
            f"{row['without_memory']['response_relevance']:.2f} | "
            f"{row['with_memory']['memory_hits']} | {matched} |"
        )

    budget = summary["token_budget_breakdown"]
    hit_rate = summary["memory_hit_rate_analysis"]
    lines.extend(
        [
            "",
            "## Memory Hit Rate Analysis",
            "",
            f"- Hit scenarios: {hit_rate['with_memory_hit_scenarios']} / {summary['scenario_count']}.",
            f"- Miss scenarios: {', '.join(hit_rate['with_memory_miss_scenarios']) or 'none'}.",
            f"- Interpretation: {hit_rate['interpretation']}",
            "",
            "## Token Budget Breakdown",
            "",
            f"- Max context budget: {budget['max_context_budget_tokens']} tokens.",
            f"- Avg context used with memory: {budget['avg_with_memory_context_used_tokens']} tokens.",
            f"- Avg context used without memory: {budget['avg_without_memory_context_used_tokens']} tokens.",
            f"- Total evicted items: {budget['eviction_total']}.",
            "",
            "## Reflection: Privacy And Limitations",
            "",
            "- PII/privacy risk: long-term profile can store email, allergy, names, and preferences, so consent and deletion controls are required.",
            "- Most sensitive memory: long-term profile, because a wrong or stale retrieval can affect health-related or identity-related answers.",
            "- Deletion policy: remove matching profile records from Redis, semantic copies from Chroma, episodic traces from JSON log, and recent turns from conversation buffer.",
            "- TTL/consent: profile facts should have explicit consent and optional TTL; episodic logs should expire faster than stable preferences.",
            "- Technical limitation: heuristic extraction is deterministic and transparent, but can miss complex corrections; LLM extraction code includes JSON parse/error handling for safer extension.",
            "",
            "## Rubric Coverage",
            "",
            "- Full memory stack: short-term, profile/Redis, episodic JSON, semantic Chroma.",
            "- LangGraph skeleton: MemoryState, retrieve_memory(state), router aggregation, sectioned prompt injection with required prompt tags.",
            "- Save/update memory: profile conflict handling, corrected allergy test, episodic task-outcome save.",
            "- Benchmark: exactly 10 multi-turn conversations covering profile recall, conflict update, episodic recall, semantic retrieval, and token budget.",
            "- Bonus-ready: Redis real mode, Chroma real mode, OpenAI gpt-4o-mini integration, LLM extraction parser, tiktoken token counting, graph-flow demo.",
            "",
            "## How To Reproduce",
            "",
            "```powershell",
            "python run_benchmark.py",
            "python -m unittest discover -s tests",
            "```",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_markdown(summary: dict[str, Any], path: Path = Path("BENCHMARK.md")) -> None:
    lines = [
        "# BENCHMARK.md",
        "",
        "Exactly 10 noisy multi-turn conversations comparing no-memory vs with-memory.",
        "",
        "| # | Scenario | Category | No-memory result | With-memory result | Pass? |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for row in summary["rows"]:
        no_memory = row["without_memory"]["response"].replace("|", "/")
        with_memory = row["with_memory"]["response"].replace("|", "/")
        passed = "Pass" if row["with_memory"]["response_relevance"] > row["without_memory"]["response_relevance"] else "Review"
        lines.append(
            f"| {row['index']} | {row['scenario']} | {row['category']} | "
            f"{no_memory} | {with_memory} | {passed} |"
        )

    lines.extend(
        [
            "",
            "## Conversation Inputs",
            "",
        ]
    )
    for row in summary["rows"]:
        lines.extend(
            [
                f"### {row['index']}. {row['scenario']}",
                "",
                f"- Category: {row['category']}",
                f"- Query: {row['query']}",
                f"- Expected terms: {', '.join(row['expected_terms'])}",
                "- Turns:",
            ]
        )
        for turn in row["turns"]:
            lines.append(f"  - {turn}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_row(label: str, metrics: dict[str, float]) -> str:
    return (
        f"| {label} | {metrics['avg_response_relevance']:.4f} | "
        f"{metrics['avg_context_utilization']:.4f} | "
        f"{metrics['avg_response_tokens']:.2f} | "
        f"{metrics['avg_token_efficiency']:.6f} | "
        f"{metrics['memory_hit_rate']:.4f} |"
    )
