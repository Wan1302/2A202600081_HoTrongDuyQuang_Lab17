# Lab 17 - Multi-Memory Agent With Benchmark

This repository implements Lab 17: a multi-memory agent with four memory backends,
router/state management, prompt injection, context trimming, and a benchmark
comparing no-memory vs with-memory agents.

## What Is Included

- Four memory backends:
  - `ConversationBufferMemory`: short-term sliding window.
  - `RedisLongTermMemory`: long-term profile/fact memory using Redis, with JSON fallback.
  - `JsonEpisodicLogMemory`: episodic JSON log.
  - `ChromaSemanticMemory`: semantic retrieval using ChromaDB, with JSON fallback.
- LangGraph-style skeleton:
  - `MemoryState`
  - `retrieve_memory(state)`
  - graph flow demo
  - sectioned memory prompt injection
- Prompt format:
  - `<persona>`
  - `<rules>`
  - `<tools_instruction>`
  - `<response_format>`
  - `<constraints>`
- OpenAI integration:
  - default model: `gpt-4o-mini`
  - default mode: use OpenAI when `OPENAI_API_KEY` exists, otherwise fallback.
- Benchmark:
  - exactly 10 noisy multi-turn conversations
  - no-memory vs with-memory comparison
  - profile recall, conflict update, episodic recall, semantic retrieval, token budget

## Clone And Setup

```powershell
git clone <your-repo-url>
cd lab_17
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` before running with real services or OpenAI.

## Quick Run Without External Services

This mode is easiest for a first check. It uses local fallbacks for Redis/Chroma
and uses rule-based generation unless an OpenAI key is present.

In `.env`, keep:

```text
LAB17_REQUIRE_REDIS=false
LAB17_REQUIRE_CHROMA=false
LAB17_EXTRACTOR=heuristic
```

Run:

```powershell
python -m unittest discover -s tests
python run_benchmark.py --llm none
```

## Run With Real Redis And ChromaDB

### 1. Start Redis

If this is the first time:

```powershell
docker run --name lab17-redis -p 6379:6379 redis:7
```

This command runs Redis in the current terminal. When you see:

```text
Ready to accept connections tcp
```

open a second PowerShell terminal for the benchmark.

If the container already exists:

```powershell
docker start lab17-redis
```

Check Redis:

```powershell
docker ps
```

### 2. Configure `.env`

Use ChromaDB local persistent mode. No separate Chroma server is needed.

```text
REDIS_URL=redis://localhost:6379/0
LAB17_REQUIRE_REDIS=true
LAB17_DISABLE_REDIS=false

CHROMA_PATH=data/chroma_db
CHROMA_HOST=
CHROMA_PORT=8000
LAB17_REQUIRE_CHROMA=true
LAB17_DISABLE_CHROMA=false

LAB17_EXTRACTOR=heuristic
```

Run:

```powershell
python run_benchmark.py --llm none
```

Expected backend modes in `reports/report.md`:

```text
redis_long_term = redis
chroma_semantic = chroma
```

## Run With OpenAI `gpt-4o-mini`

Set your key in `.env`:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o-mini
```

Run a low-cost smoke test:

```powershell
python run_benchmark.py --limit 1
```

Run the full benchmark:

```powershell
python run_benchmark.py
```

By default, `run_benchmark.py` uses OpenAI when `OPENAI_API_KEY` exists. If no
key exists, it falls back to the deterministic rule-based model.

Force OpenAI explicitly:

```powershell
python run_benchmark.py --llm openai --model gpt-4o-mini
```

Force fallback explicitly:

```powershell
python run_benchmark.py --llm none
```

Optional LLM-based memory extraction:

```text
LAB17_EXTRACTOR=openai
```

The extractor asks `gpt-4o-mini` for strict JSON memory facts and falls back to
the deterministic extractor if parsing or the API call fails.

## Commands For Grading

Recommended full run:

```powershell
.\venv\Scripts\Activate.ps1
docker start lab17-redis
python -m unittest discover -s tests
python run_benchmark.py
Get-Content reports\report.md
```

If the grader does not want to use OpenAI credits:

```powershell
python run_benchmark.py --llm none
```

## Outputs

- `BENCHMARK.md`: grading-oriented table with exactly 10 multi-turn conversations.
- `reports/report.json`: metrics table, hit-rate analysis, backend modes, prompt sections.
- `reports/report.md`: readable benchmark report, privacy reflection, rubric coverage.
- `data/`: runtime Redis/Chroma fallback or Chroma persistent data, ignored by git.

## Troubleshooting

If Redis command appears to hang, it is normal: Redis is running in foreground.
Open a second terminal or use `docker start lab17-redis` after the container has
been created.

If unit tests fail on Windows with Chroma file locks, make sure tests use the
default settings from `tests/test_lab17.py`, which disable real Redis/Chroma for
test isolation.

If benchmark fails with Redis unavailable:

```powershell
pip install redis
docker start lab17-redis
```

If benchmark fails with Chroma unavailable:

```powershell
pip install chromadb
```

If report shows `json_fallback` instead of real backends, check `.env` and rerun:

```powershell
Get-Content reports\report.md
```
