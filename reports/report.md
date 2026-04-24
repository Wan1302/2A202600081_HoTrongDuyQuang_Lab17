# Lab 17 Report

## Work Completed

- Implemented four memory backends: conversation buffer, Redis-style long-term store, JSON episodic log, and Chroma-style semantic retrieval.
- Added a MemoryState/LangGraph-style skeleton with retrieve_memory(state) and sectioned prompt injection.
- Standardized the LLM prompt with <persona>, <rules>, <tools_instruction>, <response_format>, and <constraints> blocks.
- Added profile save/update with conflict handling for corrected facts.
- Built an intent-based memory router for user preferences, factual recall, experience recall, and general context.
- Added context window management with auto-trim and four-level priority eviction.
- Added deterministic benchmark over exactly 10 noisy multi-turn conversations.

## Dataset Design

- The benchmark uses synthetic but noisy conversations with distractor facts, corrected facts, changed preferences, conditional preferences, and paraphrased queries.
- This makes the benchmark better for stress-testing routing and retrieval than a clean keyword-overlap dataset.
- The dataset is still synthetic, so the results should be treated as lab evidence rather than production-grade evaluation.

## Generation

- LLM provider: openai.
- Model: gpt-4o-mini.

## Graph Flow

`observe_turn -> save_or_update_memory -> retrieve_memory -> build_prompt_sections -> generate_answer -> save_episode_outcome`

## Memory Backends

| Backend | Mode |
| --- | --- |
| conversation_buffer | memory |
| redis_long_term | redis |
| json_episodic_log | json |
| chroma_semantic | chroma |

## Metrics Comparison

| Agent | Avg relevance | Avg context utilization | Avg response tokens | Avg token efficiency | Memory hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| With memory | 0.9600 | 0.2679 | 17.20 | 0.071360 | 1.0000 |
| Without memory | 0.1333 | 0.0615 | 16.60 | 0.005676 | 0.0000 |

## Scenario Details

| # | Category | Scenario | With-memory relevance | Without-memory relevance | Hits | Matched terms |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | profile_recall | profile_name_after_noise | 1.00 | 0.00 | 7 | linh |
| 2 | conflict_update | allergy_conflict_update | 1.00 | 0.00 | 6 | nành, đậu |
| 3 | factual_recall | rescheduled_project_deadline | 1.00 | 0.00 | 6 | 8, monday, week |
| 4 | episodic_recall | meeting_with_distractors | 1.00 | 0.00 | 8 | episodic, factual, preference, recall |
| 5 | profile_recall | conditional_language_preference | 1.00 | 0.00 | 5 | concise, english, technical, vietnamese |
| 6 | conflict_update | corrected_email_fact | 1.00 | 0.00 | 6 | edu, lab17, student, vinuni, vn |
| 7 | episodic_recall | trip_sequence_recall | 1.00 | 0.00 | 6 | before, son, tra, workshop |
| 8 | profile_recall | tool_preference_with_negative | 1.00 | 0.67 | 6 | lab, memory, pytest |
| 9 | semantic_retrieval | conflicting_library_choice | 1.00 | 0.67 | 6 | backend, chroma, semantic |
| 10 | trim_token_budget | token_budget_priority | 0.60 | 0.00 | 7 | corrected, facts, preferences |

## Memory Hit Rate Analysis

- Hit scenarios: 10 / 10.
- Miss scenarios: none.
- Interpretation: Memory improves profile recall, conflict updates, episodic recall, semantic retrieval, and token-budget cases.

## Token Budget Breakdown

- Max context budget: 187 tokens.
- Avg context used with memory: 50.1 tokens.
- Avg context used without memory: 11.5 tokens.
- Total evicted items: 0.

## Reflection: Privacy And Limitations

- PII/privacy risk: long-term profile can store email, allergy, names, and preferences, so consent and deletion controls are required.
- Most sensitive memory: long-term profile, because a wrong or stale retrieval can affect health-related or identity-related answers.
- Deletion policy: remove matching profile records from Redis, semantic copies from Chroma, episodic traces from JSON log, and recent turns from conversation buffer.
- TTL/consent: profile facts should have explicit consent and optional TTL; episodic logs should expire faster than stable preferences.
- Technical limitation: heuristic extraction is deterministic and transparent, but can miss complex corrections; LLM extraction code includes JSON parse/error handling for safer extension.

## Rubric Coverage

- Full memory stack: short-term, profile/Redis, episodic JSON, semantic Chroma.
- LangGraph skeleton: MemoryState, retrieve_memory(state), router aggregation, sectioned prompt injection with required prompt tags.
- Save/update memory: profile conflict handling, corrected allergy test, episodic task-outcome save.
- Benchmark: exactly 10 multi-turn conversations covering profile recall, conflict update, episodic recall, semantic retrieval, and token budget.
- Bonus-ready: Redis real mode, Chroma real mode, OpenAI gpt-4o-mini integration, LLM extraction parser, tiktoken token counting, graph-flow demo.

## How To Reproduce

```powershell
python run_benchmark.py
python -m unittest discover -s tests
```
