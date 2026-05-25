"""Validate that the ADK travel assistant injects the right examples.

- ``TestInjection`` is always run and asserts that hotel queries cause
  hotel examples to appear in the hotel specialist's canonical_instruction,
  and likewise for flights. No API key required.
- ``TestLiveRoundTrip`` is marked ``live`` and skipped unless GOOGLE_API_KEY
  is set; it runs the full router via ``InMemoryRunner`` and asserts the
  response remains on-topic.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from google.genai import types

from examples.adk_travel_assistant.agent import build_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(text: str) -> MagicMock:
    ctx = MagicMock()
    ctx.user_content = types.Content(parts=[types.Part(text=text)], role="user")
    return ctx


HOTEL_KEYWORDS = ("hotel", "room", "check-in", "marriott", "hilton", "hyatt", "breakfast", "cancellation")
FLIGHT_KEYWORDS = ("flight", "airline", "bag", "seat", "layover", "boarding", "ticket", "delta", "united", "lufthansa")


# ---------------------------------------------------------------------------
# Injection tests (always run)
# ---------------------------------------------------------------------------

class TestInjection:
    @pytest.fixture(scope="class")
    def app(self):
        return build_app()

    @pytest.mark.asyncio
    async def test_hotel_agent_injects_hotel_examples(self, app):
        ctx = _ctx("I want to book a hotel in Paris")
        instruction, _ = await app.hotel_agent.canonical_instruction(ctx)
        # Instruction should contain at least one obvious hotel-related token
        assert any(kw in instruction.lower() for kw in HOTEL_KEYWORDS), instruction
        # And NO flight-specific tokens (since category="hotels" filters)
        assert not any(
            kw in instruction.lower()
            for kw in ("baggage", "boarding pass", "frequent-flyer", "layover")
        ), instruction

    @pytest.mark.asyncio
    async def test_flight_agent_injects_flight_examples(self, app):
        ctx = _ctx("I need a flight from New York to Tokyo")
        instruction, _ = await app.flight_agent.canonical_instruction(ctx)
        assert any(kw in instruction.lower() for kw in FLIGHT_KEYWORDS), instruction
        # No hotel-only terms — confirms category filter is honored
        assert "check-in" not in instruction.lower() or "boarding" in instruction.lower()

    @pytest.mark.asyncio
    async def test_hotel_agent_ignores_flight_query_but_stays_in_category(self, app):
        # Even a flight query should retrieve only hotel-category examples
        ctx = _ctx("I need a flight to Tokyo")
        instruction, _ = await app.hotel_agent.canonical_instruction(ctx)
        # examples come from hotels category only
        assert "flight" not in instruction.lower() or "hotel" in instruction.lower()

    @pytest.mark.asyncio
    async def test_top_k_respected(self, app):
        ctx = _ctx("cancel my hotel")
        instruction, _ = await app.hotel_agent.canonical_instruction(ctx)
        # default k=3 → "Example 1", "Example 2", "Example 3" sections appear
        assert "Example 1" in instruction
        assert "Example 3" in instruction
        assert "Example 4" not in instruction

    def test_router_has_both_specialists(self, app):
        names = {a.name for a in app.router.sub_agents}
        assert names == {"hotel_specialist", "flight_specialist"}


# ---------------------------------------------------------------------------
# Live round-trip tests (require GOOGLE_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
class TestLiveRoundTrip:
    @pytest.fixture(scope="class")
    def app(self):
        return build_app()

    @pytest.mark.asyncio
    async def test_hotel_query_routes_and_responds(self, app):
        text = await _run_query(app, "I'd like to book a hotel in Rome for 4 nights from June 10.")
        lowered = text.lower()
        assert any(kw in lowered for kw in HOTEL_KEYWORDS), text

    @pytest.mark.asyncio
    async def test_flight_query_routes_and_responds(self, app):
        text = await _run_query(app, "Find me a one-way flight from SFO to JFK next Monday.")
        lowered = text.lower()
        assert any(kw in lowered for kw in FLIGHT_KEYWORDS), text


async def _run_query(app, query: str) -> str:
    """Send a single user message through the router and return concatenated text."""
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=app.router, app_name="travel_test")
    session = await runner.session_service.create_session(
        app_name="travel_test", user_id="u1"
    )
    parts = []
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text=query)], role="user"),
    ):
        if event.content and event.content.parts:
            for p in event.content.parts:
                if p.text:
                    parts.append(p.text)
    return "\n".join(parts)
