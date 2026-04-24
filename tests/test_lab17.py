from pathlib import Path
import os
import shutil
import unittest

from lab17.agent import MemoryAgent
from lab17.context import ContextWindowManager
from lab17.extraction import LLMExtractionParser
from lab17.memory import MemoryRecord
from lab17.router import MemoryRouter


class FakeLLM:
    provider = "fake"
    model = "fake-model"

    def generate(self, query, route, context, memory_enabled, memory_prompt=None):
        return f"fake llm answer for {route.intent}"


class Lab17Tests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LAB17_REQUIRE_REDIS"] = "false"
        os.environ["LAB17_REQUIRE_CHROMA"] = "false"
        os.environ["LAB17_DISABLE_REDIS"] = "true"
        os.environ["LAB17_DISABLE_CHROMA"] = "true"
        os.environ["LAB17_EXTRACTOR"] = "heuristic"

    def test_router_detects_preference(self) -> None:
        decision = MemoryRouter().route("What coffee do I prefer?")
        self.assertEqual(decision.intent, "user_preference")
        self.assertIn("redis_long_term", decision.backends)

    def test_context_manager_eviction_keeps_priority_one(self) -> None:
        manager = ContextWindowManager(max_tokens=30, trim_ratio=0.8)
        records = [
            MemoryRecord("important preference coffee", "preference", 1),
            MemoryRecord("low priority filler " * 20, "buffer", 4),
        ]
        result = manager.build("coffee?", records)
        kept = " ".join(item.content for item in result.items)
        self.assertIn("important preference coffee", kept)
        self.assertGreaterEqual(len(result.evicted), 1)

    def test_agent_recalls_memory(self) -> None:
        tmp = Path(".test_data/agent_memory")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            agent = MemoryAgent(memory_enabled=True, data_dir=tmp)
            agent.observe("I prefer Vietnamese iced coffee with condensed milk.")
            result = agent.answer("What coffee do I prefer?")
            self.assertGreater(result.memory_hits, 0)
            self.assertIn("Vietnamese", result.response)
            self.assertIn("redis_long_term", result.memory_backend_modes)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def test_agent_can_use_injected_llm(self) -> None:
        tmp = Path(".test_data/agent_llm")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            agent = MemoryAgent(memory_enabled=True, data_dir=tmp, llm=FakeLLM())
            result = agent.answer("What is my backup email?")
            self.assertEqual(result.llm_provider, "fake")
            self.assertEqual(result.model, "fake-model")
            self.assertIn("fake llm answer", result.response)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def test_profile_conflict_update_prefers_new_fact(self) -> None:
        tmp = Path(".test_data/profile_conflict")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            agent = MemoryAgent(memory_enabled=True, data_dir=tmp)
            agent.observe("Tôi dị ứng sữa bò.")
            agent.observe("À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.")
            result = agent.answer("Tôi dị ứng gì?")
            self.assertIn("đậu nành", result.response)
            self.assertNotIn("allergy = sữa bò", result.response)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def test_memory_state_prompt_has_required_sections(self) -> None:
        tmp = Path(".test_data/state_prompt")
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            agent = MemoryAgent(memory_enabled=True, data_dir=tmp)
            agent.observe("For lab reports, I want concise Vietnamese with technical terms kept in English.")
            result = agent.answer("How should you write my lab report summary?")
            prompt = result.memory_state["prompt"]
            self.assertIn("<persona>", prompt)
            self.assertIn("<rules>", prompt)
            self.assertIn("<tools_instruction>", prompt)
            self.assertIn("<response_format>", prompt)
            self.assertIn("<constraints>", prompt)
            self.assertIn("## User Profile", prompt)
            self.assertIn("## Episodic Memory", prompt)
            self.assertIn("## Semantic Memory", prompt)
            self.assertIn("## Recent Conversation", prompt)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def test_llm_extraction_parser_handles_bad_json(self) -> None:
        parser = LLMExtractionParser()
        self.assertEqual(parser.parse("not json"), [])
        parsed = parser.parse('{"memories":[{"key":"allergy","value":"soy"}]}')
        self.assertEqual(parsed[0].key, "allergy")
        self.assertEqual(parsed[0].value, "soy")


if __name__ == "__main__":
    unittest.main()
