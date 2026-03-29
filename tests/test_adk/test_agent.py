"""Tests for ExampleSelectorAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk.agents import LlmAgent
from google.genai import types

from example_selector.adk.agent import ExampleSelectorAgent


def _make_ctx(text: str | None = "hello"):
    ctx = MagicMock()
    if text is None:
        ctx.user_content = None
    else:
        ctx.user_content = types.Content(
            parts=[types.Part(text=text)], role="user"
        )
    return ctx


class TestExampleSelectorAgentIsLlmAgent:
    def test_is_llm_agent_subclass(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash", instruction="hi",
            example_store=store,
        )
        assert isinstance(agent, LlmAgent)

    def test_default_params(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash", instruction="hi",
            example_store=store,
        )
        assert agent.example_k == 5
        assert agent.example_category is None
        assert agent.example_format == "default"

    def test_custom_params(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash", instruction="hi",
            example_store=store,
            example_k=3,
            example_category="greeting",
        )
        assert agent.example_k == 3
        assert agent.example_category == "greeting"

    def test_native_llm_agent_params_work(self, store):
        """Native LlmAgent params (tools, output_key, etc.) are accepted."""
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash", instruction="hi",
            example_store=store,
            output_key="result",
        )
        assert agent.output_key == "result"


class TestCanonicalInstruction:
    @pytest.mark.asyncio
    async def test_appends_examples_to_instruction(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=store,
            example_k=2,
        )
        ctx = _make_ctx("greet someone")
        instruction, bypass = await agent.canonical_instruction(ctx)
        assert instruction.startswith("Be helpful.")
        assert "Example" in instruction

    @pytest.mark.asyncio
    async def test_returns_base_instruction_when_no_results(self, empty_store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=empty_store,
        )
        ctx = _make_ctx("anything")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert instruction == "Be helpful."

    @pytest.mark.asyncio
    async def test_returns_base_instruction_when_no_user_content(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=store,
        )
        ctx = _make_ctx(None)
        instruction, _ = await agent.canonical_instruction(ctx)
        assert instruction == "Be helpful."

    @pytest.mark.asyncio
    async def test_returns_base_instruction_when_empty_parts(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=store,
        )
        ctx = MagicMock()
        ctx.user_content = types.Content(parts=[], role="user")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert instruction == "Be helpful."

    @pytest.mark.asyncio
    async def test_bypass_false_for_string_instruction(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=store,
            example_k=1,
        )
        ctx = _make_ctx("hello")
        _, bypass = await agent.canonical_instruction(ctx)
        assert bypass is False

    @pytest.mark.asyncio
    async def test_callable_instruction_works(self, store):
        """Callable instructions are resolved via super() before appending examples."""
        async def dynamic_instruction(ctx):
            return "Dynamic instruction."

        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction=dynamic_instruction,
            example_store=store,
            example_k=1,
        )
        ctx = _make_ctx("hello")
        instruction, bypass = await agent.canonical_instruction(ctx)
        assert instruction.startswith("Dynamic instruction.")
        assert "Example" in instruction
        assert bypass is True  # callable => bypass=True from super()

    @pytest.mark.asyncio
    async def test_category_filter_applied(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store,
            example_k=10,
            example_category="greeting",
        )
        ctx = _make_ctx("say hello")
        instruction, _ = await agent.canonical_instruction(ctx)
        # Only greeting examples should appear
        assert "Hello" in instruction or "Goodbye" in instruction

    @pytest.mark.asyncio
    async def test_custom_format_callable(self, store):
        def my_format(examples):
            return "CUSTOM:" + ",".join(e["input_text"] for e in examples)

        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store,
            example_k=2,
            example_format=my_format,
        )
        ctx = _make_ctx("greet someone")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "CUSTOM:" in instruction


class TestExampleSelectorAgentEdgeCases:
    @pytest.mark.asyncio
    async def test_unicode_query(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store,
            example_k=2,
        )
        ctx = _make_ctx("こんにちは")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert isinstance(instruction, str)

    @pytest.mark.asyncio
    async def test_very_long_query(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store,
            example_k=2,
        )
        ctx = _make_ctx("hello " * 500)
        instruction, _ = await agent.canonical_instruction(ctx)
        assert isinstance(instruction, str)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, store):
        agent = ExampleSelectorAgent(
            name="test", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store,
            example_k=2,
        )
        ctx = _make_ctx("What's my bill? $$$")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert isinstance(instruction, str)
