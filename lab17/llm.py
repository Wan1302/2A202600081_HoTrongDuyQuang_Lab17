from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from lab17.context import ContextBuildResult
from lab17.router import RouteDecision


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(
        self,
        query: str,
        route: RouteDecision,
        context: ContextBuildResult,
        memory_enabled: bool,
        memory_prompt: str | None = None,
    ) -> str:
        ...


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class OpenAIChatLLM:
    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 160,
    ) -> None:
        load_env_file()
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your shell environment.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install the OpenAI SDK with `pip install openai`.") from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        query: str,
        route: RouteDecision,
        context: ContextBuildResult,
        memory_enabled: bool,
        memory_prompt: str | None = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Follow the tagged prompt contract exactly: <persona>, <rules>, "
                    "<tools_instruction>, <response_format>, and <constraints>. "
                    "Answer only from the provided memory context."
                ),
            },
            {
                "role": "user",
                "content": memory_prompt
                or self._build_prompt(query, route, context, memory_enabled),
            },
        ]
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        return (content or "").strip()

    def _build_prompt(
        self,
        query: str,
        route: RouteDecision,
        context: ContextBuildResult,
        memory_enabled: bool,
    ) -> str:
        context_lines = []
        for index, item in enumerate(context.items, start=1):
            context_lines.append(
                f"{index}. source={item.source}; priority={item.priority}; text={item.content}"
            )
        context_text = "\n".join(context_lines) if context_lines else "No context items."
        return (
            f"Memory enabled: {memory_enabled}\n"
            f"Detected intent: {route.intent}\n"
            f"Route reason: {route.reason}\n"
            f"Token budget used: {context.used_tokens}/{context.budget_tokens}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )


def create_llm_client(provider: str, model: str | None = None) -> LLMClient | None:
    load_env_file()
    normalized = provider.lower().strip()
    if normalized == "auto":
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIChatLLM(model=model)
        return None
    if normalized in {"none", "mock", "rule", "rule-based", ""}:
        return None
    if normalized == "openai":
        return OpenAIChatLLM(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")
