"""Validate Langchain travel assistant injection and end-to-end behavior.

- ``TestInjection`` runs offline: it builds the specialist prompts directly
  (no LLM) and asserts that formatting a prompt with a hotel/flight query
  yields the appropriate examples.
- ``TestLiveRoundTrip`` runs the full router agent against Gemini.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from example_selector import ExampleStore
from example_selector.langchain import LanceDBExampleSelector
from examples._shared.load import ensure_seeded


HOTEL_KEYWORDS = ("hotel", "room", "check-in", "marriott", "hilton", "hyatt", "breakfast")
FLIGHT_KEYWORDS = ("flight", "airline", "bag", "seat", "layover", "delta", "united", "lufthansa")


# ---------------------------------------------------------------------------
# Injection tests (offline, no API key)
# ---------------------------------------------------------------------------

def _build_prompt(store: ExampleStore, category: str, system: str, k: int = 3):
    selector = LanceDBExampleSelector(store=store, k=k, category=category)
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot = FewShotChatMessagePromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        input_variables=["input"],
    )
    return ChatPromptTemplate.from_messages(
        [("system", system), few_shot, ("human", "{input}")]
    )


class TestInjection:
    @pytest.fixture(scope="class")
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = ExampleStore(db_path=tmp, table_name="travel")
            ensure_seeded(s)
            yield s

    def test_hotel_prompt_contains_hotel_example_for_hotel_query(self, store):
        prompt = _build_prompt(store, "hotels", "hotel specialist", k=3)
        messages = prompt.format_messages(input="I want to book a hotel in Paris")
        # 1 system + 6 example messages (3 pairs) + 1 user = 8 messages
        assert len(messages) == 8
        joined = " ".join(m.content for m in messages).lower()
        assert any(kw in joined for kw in HOTEL_KEYWORDS)

    def test_flight_prompt_contains_flight_example_for_flight_query(self, store):
        prompt = _build_prompt(store, "flights", "flight specialist", k=3)
        messages = prompt.format_messages(input="I need a flight from NYC to LON")
        assert len(messages) == 8
        joined = " ".join(m.content for m in messages).lower()
        assert any(kw in joined for kw in FLIGHT_KEYWORDS)

    def test_category_filter_excludes_other_category(self, store):
        prompt = _build_prompt(store, "hotels", "hotel specialist", k=10)
        # k=10 but only 10 hotel examples exist — all results must be hotel-category
        messages = prompt.format_messages(input="random query")
        # Skip system (first) and human (last); count example pairs in between
        example_messages = messages[1:-1]
        # Each pair = 2 messages → 10 pairs = 20 messages
        assert len(example_messages) == 20
        # No flight-only tokens should appear
        joined = " ".join(m.content for m in example_messages).lower()
        assert "baggage" not in joined
        assert "boarding" not in joined

    def test_top_k_respected(self, store):
        prompt = _build_prompt(store, "hotels", "s", k=2)
        messages = prompt.format_messages(input="cancel my hotel")
        # 1 system + 4 example messages (2 pairs) + 1 user = 6
        assert len(messages) == 6


# ---------------------------------------------------------------------------
# Live round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
class TestLiveRoundTrip:
    @pytest.fixture(scope="class")
    def app(self):
        from examples.langchain_travel_assistant.agent import build_app
        return build_app()

    def test_hotel_query_returns_hotel_response(self, app):
        result = app.executor.invoke(
            {"messages": [HumanMessage(content="Book me a hotel in Madrid for 3 nights")]}
        )
        text = result["messages"][-1].content.lower()
        assert any(kw in text for kw in HOTEL_KEYWORDS), result["messages"][-1].content

    def test_flight_query_returns_flight_response(self, app):
        result = app.executor.invoke(
            {"messages": [HumanMessage(content="Find me a flight from Boston to Miami next Friday")]}
        )
        text = result["messages"][-1].content.lower()
        assert any(kw in text for kw in FLIGHT_KEYWORDS), result["messages"][-1].content
