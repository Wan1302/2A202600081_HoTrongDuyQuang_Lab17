# BENCHMARK.md

Exactly 10 noisy multi-turn conversations comparing no-memory vs with-memory.

| # | Scenario | Category | No-memory result | With-memory result | Pass? |
| ---: | --- | --- | --- | --- | --- |
| 1 | profile_name_after_noise | profile_recall | The stored memory is insufficient to determine your name. | Your name is Linh. | Pass |
| 2 | allergy_conflict_update | conflict_update | Stored memory is insufficient. | Bạn dị ứng với đậu nành. | Pass |
| 3 | rescheduled_project_deadline | factual_recall | The stored memory is insufficient to provide the deadline for the final graph RAG project. | The final graph RAG project deadline is now Monday week 8, as moved by the lecturer during class. | Pass |
| 4 | meeting_with_distractors | episodic_recall | The stored memory is insufficient to provide information about what Dr. Linh asked us to compare in the meeting. | Dr. Linh asked us to compare preference recall, factual recall, and episodic recall separately. | Pass |
| 5 | conditional_language_preference | profile_recall | The stored memory is insufficient to provide specific guidance on writing your lab report summary. | Your lab report summary should be concise in Vietnamese, with technical terms kept in English, and it should not sound like marketing copy. | Pass |
| 6 | corrected_email_fact | conflict_update | The stored memory is insufficient to provide your backup email. | Your backup email is lab17.student@vinuni.edu.vn. | Pass |
| 7 | trip_sequence_recall | episodic_recall | The stored memory is insufficient. | Before the AI workshop in Da Nang, you visited Son Tra. | Pass |
| 8 | tool_preference_with_negative | profile_recall | The stored memory is insufficient to determine which Python test tool you prefer for this memory lab. | For this memory lab, you prefer pytest because the output is easier to scan. | Pass |
| 9 | conflicting_library_choice | semantic_retrieval | The stored memory is insufficient to provide an answer regarding the semantic memory backend for the lab implementation. | The lab requirement specifically asks for Chroma as the semantic backend. | Pass |
| 10 | token_budget_priority | trim_token_budget | The stored memory is insufficient to provide an answer to your question. | When the context window is trimmed, high priority should be given to keeping user preferences and corrected facts. | Pass |

## Conversation Inputs

### 1. profile_name_after_noise

- Category: profile_recall
- Query: What is my name?
- Expected terms: linh
- Turns:
  - My name is Linh.
  - Today I am debugging a memory router.
  - The meeting room is B204, but that is not part of my profile.
  - I also mentioned another student named Minh in the hallway.

### 2. allergy_conflict_update

- Category: conflict_update
- Query: Tôi dị ứng gì?
- Expected terms: đậu, nành
- Turns:
  - Tôi dị ứng sữa bò.
  - À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
  - For dinner planning, remember the corrected allergy only.

### 3. rescheduled_project_deadline

- Category: factual_recall
- Query: When is the final graph RAG project deadline now?
- Expected terms: monday, week, 8
- Turns:
  - The graph RAG project deadline was originally Friday week 7.
  - During class the lecturer moved the graph RAG deadline to Monday week 8.
  - The Friday week 7 date is only for the presentation draft now.

### 4. meeting_with_distractors

- Category: episodic_recall
- Query: What did Dr. Linh ask us to compare in the meeting?
- Expected terms: preference, factual, episodic, recall
- Turns:
  - Yesterday I had a meeting with Dr. Linh about memory routers.
  - Before that meeting I chatted with An about UI colors, unrelated to the lab.
  - Dr. Linh asked us to compare preference recall, factual recall, and episodic recall separately.
  - The room booking number was B204, but that detail is not important.

### 5. conditional_language_preference

- Category: profile_recall
- Query: How should you write my lab report summary?
- Expected terms: concise, vietnamese, technical, english
- Turns:
  - For casual chat I like English because it helps me practice.
  - For lab reports, I want concise Vietnamese with technical terms kept in English.
  - Do not make the lab report sound like marketing copy.

### 6. corrected_email_fact

- Category: conflict_update
- Query: What is my backup email?
- Expected terms: lab17, student, vinuni, edu, vn
- Turns:
  - My backup email is student17@vinuni.edu.vn, wait that may be wrong.
  - Correction: use lab17.student@vinuni.edu.vn for backup contact.
  - The old student17 address belongs to another account.

### 7. trip_sequence_recall

- Category: episodic_recall
- Query: What place did I visit before the AI workshop in Da Nang?
- Expected terms: son, tra, before, workshop
- Turns:
  - Last time in Da Nang I visited Son Tra before the AI workshop.
  - I skipped Ba Na Hills because the weather turned bad.
  - After the workshop I wrote notes about retrieval evaluation at the hotel.

### 8. tool_preference_with_negative

- Category: profile_recall
- Query: Which Python test tool do I prefer for this memory lab?
- Expected terms: pytest, memory, lab
- Turns:
  - I tried unittest for an older Python lab.
  - For this memory lab I prefer pytest because the output is easier to scan.
  - Do not assume I want a web dashboard for the benchmark.

### 9. conflicting_library_choice

- Category: semantic_retrieval
- Query: Which semantic memory backend should the lab implementation use?
- Expected terms: chroma, semantic, backend
- Turns:
  - At first I planned to use FAISS for semantic memory.
  - The lab requirement specifically asks for Chroma as the semantic backend.
  - Use FAISS only as a comparison note, not as the implementation choice.

### 10. token_budget_priority

- Category: trim_token_budget
- Query: What should be kept first when the context window is trimmed?
- Expected terms: preferences, corrected, facts, relevant, episodic
- Turns:
  - High priority: keep user preferences and corrected facts even when trimming context.
  - Medium priority: keep directly relevant episodic memories.
  - Low priority: remove filler chatter, old jokes, and unrelated room numbers first.
  - The context window for this lab is intentionally small to test eviction.

