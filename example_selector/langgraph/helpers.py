"""Langgraph utilities for dynamic example injection."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from example_selector.core.store import ExampleStore


def create_example_prompt(
    store: ExampleStore,
    system_message: str = "",
    k: int = 5,
    category: Optional[str] = None,
):
    """Create a callable prompt for create_react_agent(prompt=...).

    The returned callable takes the graph state and returns a list of messages
    with dynamically selected examples inserted between the system message
    and the conversation history.
    """

    def _prompt(state: dict) -> list[BaseMessage]:
        messages = state.get("messages", [])

        # Find the last human message for the similarity query
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        result_messages: list[BaseMessage] = []

        if system_message:
            result_messages.append(SystemMessage(content=system_message))

        if query:
            examples = store.search(query, k=k, category=category)
            for ex in examples:
                result_messages.append(HumanMessage(content=ex["input_text"]))
                result_messages.append(AIMessage(content=ex["output_text"]))

        result_messages.extend(messages)
        return result_messages

    return _prompt


def create_example_node(
    store: ExampleStore,
    k: int = 5,
    category: Optional[str] = None,
    message_key: str = "messages",
):
    """Create a graph node function that enriches state with example messages.

    For use in custom Langgraph graphs. The node reads the latest human message,
    finds similar examples, and prepends them to the messages in state.

    Usage in a graph:
        builder.add_node("inject_examples", create_example_node(store))
        builder.add_edge(START, "inject_examples")
        builder.add_edge("inject_examples", "agent")
    """

    def _node(state: dict) -> dict:
        messages = state.get(message_key, [])

        # Find the last human message
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if not query:
            return state

        examples = store.search(query, k=k, category=category)
        if not examples:
            return state

        example_messages: list[BaseMessage] = []
        for ex in examples:
            example_messages.append(HumanMessage(content=ex["input_text"]))
            example_messages.append(AIMessage(content=ex["output_text"]))

        return {message_key: example_messages + list(messages)}

    return _node
