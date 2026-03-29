"""Integration tests for the ADK adapter.

Tests that ExampleSelectorAgent, ExampleSelectorPlugin, and
create_example_callback work together with the core ExampleStore.
"""

import pytest

from google.adk.agents import LlmAgent
from google.genai import types
from unittest.mock import MagicMock

from example_selector.adk import (
    ExampleSelectorAgent,
    ExampleSelectorPlugin,
    create_example_callback,
)


class TestExampleSelectorAgentIntegration:
    """ExampleSelectorAgent integrates with ExampleStore for end-to-end injection."""

    @pytest.mark.asyncio
    async def test_examples_appear_in_instruction(self, store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="You are helpful.",
            example_store=store, example_k=2,
        )
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="greet someone")], role="user"
        )
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "You are helpful." in instruction
        assert "Example" in instruction

    @pytest.mark.asyncio
    async def test_relevant_examples_selected(self, store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store, example_k=2,
        )
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="calculate sum of numbers")], role="user"
        )
        instruction, _ = await agent.canonical_instruction(ctx)
        # The math example should be semantically closest
        assert "sum" in instruction.lower() or "2" in instruction or "3" in instruction

    @pytest.mark.asyncio
    async def test_used_as_sub_agent(self, store):
        """ExampleSelectorAgent is accepted anywhere LlmAgent is accepted."""
        child = ExampleSelectorAgent(
            name="child", model="gemini-2.0-flash",
            instruction="handle math",
            example_store=store,
            example_category="math",
        )
        parent = LlmAgent(
            name="parent", model="gemini-2.0-flash",
            instruction="route requests",
            sub_agents=[child],
        )
        assert parent.sub_agents[0] is child

    @pytest.mark.asyncio
    async def test_runtime_store_update_visible(self, store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=store, example_k=3,
        )
        store.add_example(
            input_text="What is the speed of light?",
            output_text="The speed of light is approximately 299,792,458 m/s.",
            category="physics",
        )
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="speed of light")], role="user"
        )
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "light" in instruction.lower() or "299" in instruction


class TestExampleSelectorPluginIntegration:
    @pytest.mark.asyncio
    async def test_plugin_injects_examples(self, store):
        plugin = ExampleSelectorPlugin(store=store, k=2)
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="greet someone")], role="user"
        )
        req = MagicMock()
        req.append_instructions = MagicMock()
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        req.append_instructions.assert_called_once()

    @pytest.mark.asyncio
    async def test_plugin_no_op_on_empty_store(self, empty_store):
        plugin = ExampleSelectorPlugin(store=empty_store)
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="anything")], role="user"
        )
        req = MagicMock()
        req.append_instructions = MagicMock()
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        req.append_instructions.assert_not_called()


class TestCallbackIntegration:
    def test_callback_injected_into_agent(self, store):
        callback = create_example_callback(store, k=2)
        agent = LlmAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="Be helpful.",
            before_model_callback=callback,
        )
        assert agent.before_model_callback is callback
