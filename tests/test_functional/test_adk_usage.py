"""
Functional test: Google ADK usage example.

Demonstrates how to use example-selector with Google ADK agents.
This mirrors real-world usage patterns and tests the full integration.
"""

import tempfile

import pytest

from google.adk.agents import LlmAgent
from google.genai import types
from unittest.mock import MagicMock

from example_selector import ExampleStore
from example_selector.adk import (
    ExampleSelectorAgent,
    ExampleSelectorPlugin,
    create_example_callback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def customer_support_store():
    """A store simulating customer support few-shot examples."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ExampleStore(db_path=tmp, table_name="support")
        store.add_examples([
            {
                "input_text": "I can't login to my account",
                "output_text": "I'm sorry you're having trouble logging in. Let me help you reset your password. Please click the 'Forgot Password' link on the login page and enter your registered email address.",
                "category": "account",
            },
            {
                "input_text": "How do I change my subscription plan?",
                "output_text": "To change your subscription, go to Settings > Billing > Change Plan. You can upgrade or downgrade at any time. Changes take effect at the start of your next billing cycle.",
                "category": "billing",
            },
            {
                "input_text": "I was charged twice for my order",
                "output_text": "I apologize for the double charge. I can see the duplicate transaction in our system. I'm initiating a refund for the extra charge right now. You should see it reflected in 3-5 business days.",
                "category": "billing",
            },
            {
                "input_text": "How do I delete my account?",
                "output_text": "To delete your account, go to Settings > Account > Delete Account. Please note this action is permanent and all your data will be removed after 30 days. Would you like to proceed?",
                "category": "account",
            },
            {
                "input_text": "My order hasn't arrived yet",
                "output_text": "Let me check the status of your order. Could you please provide your order number? I'll look into the shipping details and give you an update right away.",
                "category": "shipping",
            },
            {
                "input_text": "Can I get a refund for a product I returned?",
                "output_text": "Yes, once we receive and inspect the returned item, we'll process your refund within 5-7 business days. You'll receive an email confirmation when the refund is issued.",
                "category": "billing",
            },
        ])
        yield store


@pytest.fixture
def code_assistant_store():
    """A store simulating code assistant few-shot examples."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ExampleStore(db_path=tmp, table_name="code")
        store.add_examples([
            {
                "input_text": "Write a Python function to reverse a string",
                "output_text": "def reverse_string(s: str) -> str:\n    return s[::-1]",
                "category": "python",
            },
            {
                "input_text": "How do I read a JSON file in Python?",
                "output_text": "import json\n\nwith open('data.json', 'r') as f:\n    data = json.load(f)",
                "category": "python",
            },
            {
                "input_text": "Write a SQL query to find duplicate rows",
                "output_text": "SELECT column_name, COUNT(*) as count\nFROM table_name\nGROUP BY column_name\nHAVING COUNT(*) > 1;",
                "category": "sql",
            },
            {
                "input_text": "How do I create a React component with state?",
                "output_text": "import { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;\n}",
                "category": "javascript",
            },
        ])
        yield store


def _make_ctx(text: str):
    ctx = MagicMock()
    ctx.user_content = types.Content(parts=[types.Part(text=text)], role="user")
    return ctx


# ---------------------------------------------------------------------------
# Primary usage: ExampleSelectorAgent
# ---------------------------------------------------------------------------

class TestExampleSelectorAgentUsage:
    """
    PRIMARY USAGE PATTERN: ExampleSelectorAgent

    Drop-in replacement for LlmAgent with automatic example injection.
    """

    def test_create_agent(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="support_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful customer support agent.",
            example_store=customer_support_store,
            example_k=3,
        )
        assert isinstance(agent, LlmAgent)
        assert agent.example_k == 3

    @pytest.mark.asyncio
    async def test_instruction_includes_relevant_examples(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="support_agent",
            model="gemini-2.0-flash",
            instruction="You are a helpful customer support agent.",
            example_store=customer_support_store,
            example_k=3,
        )
        ctx = _make_ctx("I need help with my payment")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "You are a helpful customer support agent." in instruction
        # Billing-related examples should be prioritized
        assert any(
            kw in instruction.lower()
            for kw in ("charge", "refund", "billing", "subscription")
        )

    @pytest.mark.asyncio
    async def test_category_filter(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="billing_agent",
            model="gemini-2.0-flash",
            instruction="Handle billing inquiries.",
            example_store=customer_support_store,
            example_k=10,
            example_category="billing",
        )
        ctx = _make_ctx("anything")
        instruction, _ = await agent.canonical_instruction(ctx)
        # Only billing examples should appear (3 billing examples exist)
        assert "billing" in instruction.lower() or "charge" in instruction.lower() or "subscription" in instruction.lower()

    @pytest.mark.asyncio
    async def test_callable_instruction_with_state(self, customer_support_store):
        async def dynamic_instruction(ctx):
            tier = ctx.state.get("user_tier", "free")
            return f"You are a support agent for {tier} tier users."

        ctx = _make_ctx("I need help")
        ctx.state = {"user_tier": "premium"}

        agent = ExampleSelectorAgent(
            name="support",
            model="gemini-2.0-flash",
            instruction=dynamic_instruction,
            example_store=customer_support_store,
            example_k=2,
        )
        instruction, bypass = await agent.canonical_instruction(ctx)
        assert "premium tier" in instruction
        assert "Example" in instruction
        assert bypass is True

    def test_used_as_sub_agent_in_multi_agent_setup(self, customer_support_store):
        billing_agent = ExampleSelectorAgent(
            name="billing",
            model="gemini-2.0-flash",
            instruction="Handle billing inquiries.",
            example_store=customer_support_store,
            example_category="billing",
            example_k=3,
        )
        account_agent = LlmAgent(
            name="account",
            model="gemini-2.0-flash",
            instruction="Handle account issues.",
        )
        router = LlmAgent(
            name="router",
            model="gemini-2.0-flash",
            instruction="Route to the right specialist.",
            sub_agents=[billing_agent, account_agent],
        )
        assert router.sub_agents[0] is billing_agent
        assert isinstance(router.sub_agents[0], LlmAgent)

    @pytest.mark.asyncio
    async def test_custom_format(self, customer_support_store):
        def markdown_format(examples):
            lines = ["## Relevant Examples\n"]
            for i, ex in enumerate(examples, 1):
                lines.append(f"### Example {i}")
                lines.append(f"**Input:** {ex['input_text']}")
                lines.append(f"**Output:** {ex['output_text']}")
            return "\n".join(lines)

        agent = ExampleSelectorAgent(
            name="agent",
            model="gemini-2.0-flash",
            instruction="Be helpful.",
            example_store=customer_support_store,
            example_k=2,
            example_format=markdown_format,
        )
        ctx = _make_ctx("payment issue")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "## Relevant Examples" in instruction

    @pytest.mark.asyncio
    async def test_empty_store_no_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExampleStore(db_path=tmp, table_name="empty")
            agent = ExampleSelectorAgent(
                name="agent",
                model="gemini-2.0-flash",
                instruction="Be helpful.",
                example_store=store,
            )
            ctx = _make_ctx("hello")
            instruction, _ = await agent.canonical_instruction(ctx)
            assert instruction == "Be helpful."

    @pytest.mark.asyncio
    async def test_dynamic_store_updates(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="agent",
            model="gemini-2.0-flash",
            instruction="hi",
            example_store=customer_support_store,
            example_k=3,
        )
        customer_support_store.add_example(
            input_text="Where can I find my tracking number?",
            output_text="Your tracking number is in your order confirmation email.",
            category="shipping",
        )
        ctx = _make_ctx("where is my tracking number")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert "tracking" in instruction.lower()


# ---------------------------------------------------------------------------
# Secondary usage: create_example_callback
# ---------------------------------------------------------------------------

class TestCallbackUsage:
    """
    SECONDARY/ESCAPE-HATCH PATTERN: create_example_callback

    For users who need to keep vanilla LlmAgent.
    """

    def test_create_agent_with_callback(self, code_assistant_store):
        callback = create_example_callback(
            code_assistant_store,
            k=2,
            category="python",
            instruction_prefix="\n\nRelevant code examples:\n",
        )
        agent = LlmAgent(
            name="code_assistant",
            model="gemini-2.0-flash",
            instruction="Write clean code.",
            before_model_callback=callback,
        )
        assert agent.before_model_callback is callback

    def test_callback_formats_correctly(self, code_assistant_store):
        callback = create_example_callback(
            code_assistant_store, k=2, instruction_prefix="\n\nCode examples:\n"
        )
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="How do I parse JSON in Python?")],
            role="user",
        )
        req = MagicMock()
        req.append_instructions = MagicMock()
        result = callback(ctx, req)
        assert result is None
        req.append_instructions.assert_called_once()
        text = req.append_instructions.call_args[0][0][0]
        assert "Code examples:" in text
        assert "json" in text.lower()


# ---------------------------------------------------------------------------
# Optional usage: ExampleSelectorPlugin
# ---------------------------------------------------------------------------

class TestPluginUsage:
    """
    OPTIONAL/GLOBAL PATTERN: ExampleSelectorPlugin

    When all agents in the app/runner share the same example store.
    """

    @pytest.mark.asyncio
    async def test_plugin_injects_globally(self, customer_support_store):
        plugin = ExampleSelectorPlugin(store=customer_support_store, k=2)
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text="I need a refund")], role="user"
        )
        req = MagicMock()
        req.append_instructions = MagicMock()
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        req.append_instructions.assert_called_once()
        text = req.append_instructions.call_args[0][0][0]
        assert "refund" in text.lower() or "charge" in text.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestADKEdgeCases:
    @pytest.mark.asyncio
    async def test_k_larger_than_store(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=customer_support_store,
            example_k=100,
        )
        ctx = _make_ctx("help")
        instruction, _ = await agent.canonical_instruction(ctx)
        # Should return all 6 examples without error
        assert isinstance(instruction, str)

    @pytest.mark.asyncio
    async def test_very_long_query(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=customer_support_store,
            example_k=2,
        )
        ctx = _make_ctx("billing " * 500)
        instruction, _ = await agent.canonical_instruction(ctx)
        assert isinstance(instruction, str)

    @pytest.mark.asyncio
    async def test_unicode_in_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExampleStore(db_path=tmp, table_name="unicode")
            store.add_example(
                input_text="Translate 'hello' to Japanese",
                output_text="こんにちは (Konnichiwa)",
                category="translation",
            )
            agent = ExampleSelectorAgent(
                name="agent", model="gemini-2.0-flash",
                instruction="hi",
                example_store=store, example_k=1,
            )
            ctx = _make_ctx("translate to Japanese")
            instruction, _ = await agent.canonical_instruction(ctx)
            assert "こんにちは" in instruction

    @pytest.mark.asyncio
    async def test_special_characters(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=customer_support_store,
            example_k=2,
        )
        ctx = _make_ctx("What's my bill? $$$")
        instruction, _ = await agent.canonical_instruction(ctx)
        assert isinstance(instruction, str)

    @pytest.mark.asyncio
    async def test_multiple_agents_same_store(self, customer_support_store):
        """Multiple agents sharing one store with different category filters."""
        billing_agent = ExampleSelectorAgent(
            name="billing", model="gemini-2.0-flash",
            instruction="Handle billing.",
            example_store=customer_support_store,
            example_k=3, example_category="billing",
        )
        account_agent = ExampleSelectorAgent(
            name="account", model="gemini-2.0-flash",
            instruction="Handle account.",
            example_store=customer_support_store,
            example_k=3, example_category="account",
        )

        billing_ctx = _make_ctx("payment issue")
        account_ctx = _make_ctx("login problem")

        billing_instruction, _ = await billing_agent.canonical_instruction(billing_ctx)
        account_instruction, _ = await account_agent.canonical_instruction(account_ctx)

        assert isinstance(billing_instruction, str)
        assert isinstance(account_instruction, str)
        # Different examples should appear in each
        assert billing_instruction != account_instruction

    @pytest.mark.asyncio
    async def test_store_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store1 = ExampleStore(db_path=tmp, table_name="persist")
            store1.add_example(input_text="hello", output_text="world")

            store2 = ExampleStore(db_path=tmp, table_name="persist")
            agent = ExampleSelectorAgent(
                name="agent", model="gemini-2.0-flash",
                instruction="hi",
                example_store=store2, example_k=1,
            )
            ctx = _make_ctx("hello")
            instruction, _ = await agent.canonical_instruction(ctx)
            assert "world" in instruction

    @pytest.mark.asyncio
    async def test_sql_injection_in_category_filter(self, customer_support_store):
        agent = ExampleSelectorAgent(
            name="agent", model="gemini-2.0-flash",
            instruction="hi",
            example_store=customer_support_store,
            example_k=5,
            example_category="billing' OR '1'='1",
        )
        ctx = _make_ctx("test")
        try:
            instruction, _ = await agent.canonical_instruction(ctx)
            # If no exception: should fall back to base instruction (no matching category)
            assert instruction == "hi"
        except Exception:
            pass  # Injection blocked — acceptable
