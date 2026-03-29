"""Functional integration tests for Langgraph adapter.

Tests that our helpers integrate correctly with Langgraph's
create_react_agent and custom graph workflows.
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from example_selector.langgraph.helpers import create_example_prompt, create_example_node


class TestCreateReactAgentIntegration:
    """Test that create_example_prompt produces valid input for create_react_agent."""

    def test_prompt_output_is_valid_language_model_input(self, store):
        """The callable must return list[BaseMessage] which is valid LanguageModelInput."""
        prompt = create_example_prompt(store, system_message="You are helpful.", k=2)
        state = {"messages": [HumanMessage(content="hello")]}
        result = prompt(state)
        # Must be a list of BaseMessage instances
        from langchain_core.messages import BaseMessage

        assert isinstance(result, list)
        assert all(isinstance(m, BaseMessage) for m in result)

    def test_prompt_preserves_conversation_history(self, store):
        """The prompt should preserve the full conversation history."""
        prompt = create_example_prompt(store, system_message="System.", k=1)
        history = [
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="second question about math"),
        ]
        state = {"messages": history}
        result = prompt(state)
        # All original messages should be at the end
        assert result[-3:] == history

    def test_prompt_with_relevance(self, store):
        """Examples should be semantically relevant to the query."""
        prompt = create_example_prompt(store, k=1)
        state = {"messages": [HumanMessage(content="translate to Spanish")]}
        result = prompt(state)
        # Should include translation example
        example_texts = [m.content for m in result if m.content != "translate to Spanish"]
        assert any(
            "translate" in t.lower() or "hola" in t.lower() or "spanish" in t.lower()
            for t in example_texts
        )


class TestCustomGraphIntegration:
    """Test create_example_node for custom Langgraph graphs."""

    def test_node_output_compatible_with_add_messages(self, store):
        """The node output should be compatible with Langgraph's add_messages reducer."""
        node = create_example_node(store, k=2)
        state = {"messages": [HumanMessage(content="greet someone")]}
        result = node(state)

        # Result should have messages key with list of BaseMessage
        from langchain_core.messages import BaseMessage

        assert "messages" in result
        assert all(isinstance(m, BaseMessage) for m in result["messages"])

    def test_node_examples_before_originals(self, store):
        """Examples should come before original messages."""
        node = create_example_node(store, k=1)
        original = HumanMessage(content="hello")
        state = {"messages": [original]}
        result = node(state)
        msgs = result["messages"]
        # Original should be last
        assert msgs[-1] is original
        # Examples should be before
        assert len(msgs) > 1

    def test_full_pipeline_simulation(self, store):
        """Simulate a full graph pipeline: inject examples -> agent would process."""
        # Step 1: Create initial state
        state = {"messages": [HumanMessage(content="calculate 2 plus 3")]}

        # Step 2: Run example injection node
        inject_node = create_example_node(store, k=2)
        enriched_state = inject_node(state)

        # Step 3: Verify enriched state is valid for an LLM node
        msgs = enriched_state["messages"]
        assert len(msgs) >= 3  # At least 1 example pair + original
        # Last message is the user query
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "calculate 2 plus 3"
        # Examples are human+ai pairs
        for i in range(0, len(msgs) - 1, 2):
            if i + 1 < len(msgs) - 1:  # Don't check the last original message
                assert isinstance(msgs[i], HumanMessage)
                assert isinstance(msgs[i + 1], AIMessage)

    def test_multiple_invocations_independent(self, store):
        """Each invocation should be independent (no state leakage)."""
        node = create_example_node(store, k=1)
        state1 = {"messages": [HumanMessage(content="greet someone")]}
        state2 = {"messages": [HumanMessage(content="calculate math")]}

        result1 = node(state1)
        result2 = node(state2)

        # Results should be different (different queries)
        assert result1["messages"][0].content != result2["messages"][0].content or \
               result1["messages"][1].content != result2["messages"][1].content
