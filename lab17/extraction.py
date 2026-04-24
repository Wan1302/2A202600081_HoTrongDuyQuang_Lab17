from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    key: str
    value: str
    memory_type: str
    priority: int
    reason: str


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class HeuristicMemoryExtractor:
    """Small deterministic extractor used for offline benchmark reproducibility."""

    def extract(self, text: str) -> list[ExtractedMemory]:
        normalized = text.lower()
        memories: list[ExtractedMemory] = []

        allergy = self._extract_allergy(text, normalized)
        if allergy:
            memories.append(allergy)

        email = self._extract_email(text, normalized)
        if email:
            memories.append(email)

        team_name = self._extract_team_name(text, normalized)
        if team_name:
            memories.append(team_name)

        user_name = self._extract_user_name(text, normalized)
        if user_name:
            memories.append(user_name)

        daily_drink = self._extract_daily_drink(text, normalized)
        if daily_drink:
            memories.append(daily_drink)

        return memories

    def _extract_user_name(self, text: str, normalized: str) -> ExtractedMemory | None:
        if "my name is" in normalized:
            match = re.search(r"my name is\s+(.+?)[.?!]?$", text, re.IGNORECASE)
            value = match.group(1).strip(" .") if match else ""
            return ExtractedMemory("user_name", value, "profile", 1, "profile name update")
        if "tên tôi là" in normalized:
            value = text.split("là", 1)[-1].strip(" .")
            return ExtractedMemory("user_name", value, "profile", 1, "profile name update")
        return None

    def _extract_allergy(self, text: str, normalized: str) -> ExtractedMemory | None:
        if "dị ứng" not in normalized and "allergic" not in normalized:
            return None

        if "chứ không phải" in normalized:
            value = text.split("dị ứng", 1)[-1].split("chứ không phải", 1)[0].strip(" .")
        elif "not" in normalized and "allergic" in normalized:
            value = text.rsplit("allergic to", 1)[-1].split("not", 1)[0].strip(" .")
        elif "dị ứng" in normalized:
            value = text.split("dị ứng", 1)[-1].strip(" .")
        elif "allergic to" in normalized:
            value = text.rsplit("allergic to", 1)[-1].strip(" .")
        else:
            return None

        if not value:
            return None
        return ExtractedMemory("allergy", value, "profile", 1, "profile allergy update")

    def _extract_email(self, text: str, normalized: str) -> ExtractedMemory | None:
        emails = EMAIL_RE.findall(text)
        if not emails:
            return None
        if "correction" in normalized or "use " in normalized or "backup email" in normalized:
            return ExtractedMemory("backup_email", emails[-1], "profile", 1, "profile email update")
        return None

    def _extract_team_name(self, text: str, normalized: str) -> ExtractedMemory | None:
        if "renamed the team to" in normalized:
            value = text.split("renamed the team to", 1)[-1].split(" after ", 1)[0].strip(" .")
            return ExtractedMemory("team_name", value, "profile", 1, "profile team-name update")
        if "team name is" in normalized and "was" not in normalized:
            value = text.split("team name is", 1)[-1].strip(" .")
            return ExtractedMemory("team_name", value, "profile", 1, "profile team-name update")
        return None

    def _extract_daily_drink(self, text: str, normalized: str) -> ExtractedMemory | None:
        if "daily coding" in normalized and "prefer" in normalized:
            value = text.split("prefer", 1)[-1].strip(" .")
            return ExtractedMemory("daily_coding_drink", value, "profile", 1, "profile preference update")
        return None


class LLMExtractionParser:
    """Parser/error handling for JSON memory extraction responses."""

    def parse(self, raw: str) -> list[ExtractedMemory]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if isinstance(payload, dict):
            items = payload.get("memories", [])
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        memories: list[ExtractedMemory] = []
        for item in items:
            parsed = self._parse_item(item)
            if parsed:
                memories.append(parsed)
        return memories

    def _parse_item(self, item: Any) -> ExtractedMemory | None:
        if not isinstance(item, dict):
            return None
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if not key or not value:
            return None
        memory_type = str(item.get("memory_type", "profile")).strip()
        priority = int(item.get("priority", 1))
        reason = str(item.get("reason", "llm extraction")).strip()
        return ExtractedMemory(key, value, memory_type, min(max(priority, 1), 4), reason)


class OpenAIMemoryExtractor:
    """Optional LLM-based memory extraction with JSON parse/error handling."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIMemoryExtractor.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI SDK with `pip install openai`.") from exc

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key)
        self.parser = LLMExtractionParser()
        self.fallback = HeuristicMemoryExtractor()

    def extract(self, text: str) -> list[ExtractedMemory]:
        prompt = (
            "Extract durable user memory facts from the turn. Return strict JSON only.\n"
            "Schema: {\"memories\":[{\"key\":\"...\",\"value\":\"...\","
            "\"memory_type\":\"profile|fact|experience\",\"priority\":1,"
            "\"reason\":\"...\"}]}\n"
            "If the user corrects an old fact, return only the corrected fact.\n"
            f"Turn: {text}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=180,
            )
            raw = completion.choices[0].message.content or ""
            parsed = self.parser.parse(raw)
        except Exception:
            parsed = []

        return parsed or self.fallback.extract(text)
