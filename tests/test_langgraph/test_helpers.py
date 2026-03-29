"""Tests for Langgraph helper functions."""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from example_selector.langgraph.helpers import create_example_prompt, create_example_node


class TestCreateExamplePrompt:
    def test_returns_callable(self, store):
        prompt = create_example_prompt(store, system_message="You are helpful.")
        assert callable(prompt)

    def test_prompt_returns_messages(self, store):
        prompt = create_example_prompt(store, system_message="You are helpful.", k=2)
        state = {"messages": [HumanMessage(content="greet someone")]}
        result = prompt(state)
        assert isinstance(result, list)
        assert all(isinstance(m, (SystemMessage, HumanMessage, AIMessage)) for m in result)

    def test_prompt_starts_with_system_message(self, store):
        prompt = create_example_prompt(store, system_message="You are helpful.", k=1)
        state = {"messages": [HumanMessage(content="hello")]}
        result = prompt(state)
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are helpful."

    def test_prompt_ends_with_original_messages(self, store):
        prompt = create_example_prompt(store, system_message="System.", k=1)
        original = HumanMessage(content="hello")
        state = {"messages": [original]}
        result = prompt(state)
        assert result[-1] is original

    def test_prompt_includes_examples(self, store):
        prompt = create_example_prompt(store, k=2)
        state = {"messages": [HumanMessage(content="greet someone")]}
        result = prompt(state)
        # Should have example pairs (human+ai) plus original message
        assert len(result) >= 3  # At least 1 example pair + original

    def test_prompt_no_system_message(self, store):
        prompt = create_example_prompt(store, k=1)
        state = {"messages": [HumanMessage(content="hello")]}
        result = prompt(state)
        # First message should NOT be system (no system_message provided)
        assert not isinstance(result[0], SystemMessage)

    def test_prompt_category_filter(self, store):
        prompt = create_example_prompt(store, k=10, category="greeting")
        state = {"messages": [HumanMessage(content="anything")]}
        result = prompt(state)
        # Original message + at most 2 greeting examples (4 msgs)
        assert len(result) <= 5  # 4 example msgs + 1 original

    def test_prompt_empty_store(self, empty_store):
        prompt = create_example_prompt(empty_store, system_message="System.", k=5)
        state = {"messages": [HumanMessage(content="hello")]}
        result = prompt(state)
        # Just system + original
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)

    def test_prompt_empty_messages(self, store):
        prompt = create_example_prompt(store, system_message="System.", k=2)
        state = {"messages": []}
        result = prompt(state)
        # Just system message, no query to search with
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_prompt_multiple_messages_uses_last_human(self, store):
        prompt = create_example_prompt(store, k=1)
        state = {
            "messages": [
                HumanMessage(content="previous message"),
                AIMessage(content="response"),
                HumanMessage(content="calculate math sum"),
            ]
        }
        result = prompt(state)
        # Should use "calculate math sum" as query
        # The examples should be relevant to math
        assert len(result) >= 4  # At least 1 example pair + 3 original


class TestCreateExampleNode:
    def test_returns_callable(self, store):
        node = create_example_node(store)
        assert callable(node)

    def test_node_returns_state(self, store):
        node = create_example_node(store, k=2)
        state = {"messages": [HumanMessage(content="greet someone")]}
        result = node(state)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_node_prepends_examples(self, store):
        node = create_example_node(store, k=2)
        original = HumanMessage(content="hello")
        state = {"messages": [original]}
        result = node(state)
        messages = result["messages"]
        # Examples come first, then original
        assert messages[-1] is original
        assert len(messages) >= 3  # At least 1 example pair + original

    def test_node_example_pairs(self, store):
        node = create_example_node(store, k=1)
        state = {"messages": [HumanMessage(content="hello")]}
        result = node(state)
        messages = result["messages"]
        # First pair should be human+ai
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_node_category_filter(self, store):
        node = create_example_node(store, k=10, category="math")
        state = {"messages": [HumanMessage(content="anything")]}
        result = node(state)
        # Only 1 math example in sample
        assert len(result["messages"]) <= 3

    def test_node_empty_store(self, empty_store):
        node = create_example_node(empty_store, k=5)
        original = HumanMessage(content="hello")
        state = {"messages": [original]}
        result = node(state)
        # Should return state unchanged
        assert result is state

    def test_node_no_human_message(self, store):
        node = create_example_node(store, k=2)
        state = {"messages": [AIMessage(content="ai response")]}
        result = node(state)
        assert result is state

    def test_node_empty_messages(self, store):
        node = create_example_node(store, k=2)
        state = {"messages": []}
        result = node(state)
        assert result is state

    def test_node_custom_message_key(self, store):
        node = create_example_node(store, k=1, message_key="chat_history")
        state = {"chat_history": [HumanMessage(content="hello")]}
        result = node(state)
        assert "chat_history" in result
