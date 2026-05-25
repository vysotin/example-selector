"""Validate Langgraph travel assistant injection and end-to-end behavior."""

from __future__ import annotations

import os
import tempfile

import pytest

from langchain_core.messages import HumanMessage

from example_selector import ExampleStore
from example_selector.langgraph import create_example_prompt
from examples._shared.load import ensure_seeded


HOTEL_KEYWORDS = ("hotel", "room", "check-in", "marriott", "hilton", "hyatt", "breakfast")
FLIGHT_KEYWORDS = ("flight", "airline", "bag", "seat", "layover", "delta", "united", "lufthansa")


# ---------------------------------------------------------------------------
# Injection tests (offline)
# ---------------------------------------------------------------------------

class TestInjection:
    @pytest.fixture(scope="class")
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = ExampleStore(db_path=tmp, table_name="travel")
            ensure_seeded(s)
            yield s

    def test_hotel_prompt_includes_hotel_examples_for_hotel_query(self, store):
        prompt_fn = create_example_prompt(
            store, system_message="hotel specialist", k=3, category="hotels"
        )
        state = {"messages": [HumanMessage(content="I want to book a hotel in Paris")]}
        messages = prompt_fn(state)
        # system + 3 pairs + original user = 8 messages
        assert len(messages) == 8
        joined = " ".join(str(m.content) for m in messages).lower()
        assert any(kw in joined for kw in HOTEL_KEYWORDS)

    def test_flight_prompt_includes_flight_examples_for_flight_query(self, store):
        prompt_fn = create_example_prompt(
            store, system_message="flight specialist", k=3, category="flights"
        )
        state = {"messages": [HumanMessage(content="I need a flight from NYC to LON")]}
        messages = prompt_fn(state)
        assert len(messages) == 8
        joined = " ".join(str(m.content) for m in messages).lower()
        assert any(kw in joined for kw in FLIGHT_KEYWORDS)

    def test_hotel_prompt_filters_out_flight_examples(self, store):
        prompt_fn = create_example_prompt(
            store, system_message="hotel specialist", k=10, category="hotels"
        )
        state = {"messages": [HumanMessage(content="random query")]}
        messages = prompt_fn(state)
        # 1 system + 20 example messages (10 pairs) + 1 user = 22
        assert len(messages) == 22
        example_messages = messages[1:-1]
        joined = " ".join(str(m.content) for m in example_messages).lower()
        assert "baggage" not in joined
        assert "boarding" not in joined

    def test_top_k_respected(self, store):
        prompt_fn = create_example_prompt(
            store, system_message="s", k=2, category="hotels"
        )
        state = {"messages": [HumanMessage(content="cancel my hotel")]}
        messages = prompt_fn(state)
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
        from examples.langgraph_travel_assistant.agent import build_app
        return build_app()

    def test_hotel_query_routes_to_hotel_specialist(self, app):
        result = app.graph.invoke({
            "messages": [HumanMessage(content="I want to book a hotel in Vienna")],
            "intent": "",
        })
        assert result["intent"] == "hotels"
        final = result["messages"][-1].content.lower()
        assert any(kw in final for kw in HOTEL_KEYWORDS), result["messages"][-1].content

    def test_flight_query_routes_to_flight_specialist(self, app):
        result = app.graph.invoke({
            "messages": [HumanMessage(content="Find me a flight from Paris to Berlin tomorrow")],
            "intent": "",
        })
        assert result["intent"] == "flights"
        final = result["messages"][-1].content.lower()
        assert any(kw in final for kw in FLIGHT_KEYWORDS), result["messages"][-1].content
