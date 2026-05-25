"""ADK travel-booking assistant: router with hotel + flight specialist subagents.

Each specialist is an ``ExampleSelectorAgent`` filtered to its own category
('hotels' or 'flights') so the right multi-shot examples are injected
automatically based on the user's intent.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Optional

from google.adk.agents import LlmAgent

from example_selector import ExampleStore
from example_selector.adk import ExampleSelectorAgent
from examples._shared.load import ensure_seeded


MODEL = "gemini-2.0-flash"

HOTEL_INSTRUCTION = (
    "You are a hotel booking specialist. You help travellers search, book, "
    "modify, and cancel hotel reservations. Always confirm dates, location, "
    "and guest count before suggesting options. Be concise and follow the "
    "tone of the provided examples."
)

FLIGHT_INSTRUCTION = (
    "You are a flight booking specialist. You help travellers search flights, "
    "book tickets, manage bookings, and answer baggage and seat questions. "
    "Always confirm origin, destination, dates, and passenger count before "
    "presenting options. Be concise and mirror the style of the examples."
)

ROUTER_INSTRUCTION = (
    "You are a travel coordinator. The user wants help with travel planning. "
    "Route requests about hotels, rooms, check-in, or accommodations to "
    "hotel_specialist. Route requests about flights, airlines, seats, "
    "baggage, layovers, or boarding to flight_specialist. If a request "
    "spans both, handle the dominant intent first."
)


@dataclass
class TravelApp:
    """Container for the assembled travel-booking app."""

    router: LlmAgent
    hotel_agent: ExampleSelectorAgent
    flight_agent: ExampleSelectorAgent
    store: ExampleStore


def build_app(db_path: Optional[str] = None) -> TravelApp:
    """Build the router + 2 specialist subagents and seed the example store.

    Args:
        db_path: Filesystem path for LanceDB. If None, uses a fresh temp dir
            so each call is hermetic (handy for tests).
    """
    db_path = db_path or tempfile.mkdtemp(prefix="travel_examples_db_")
    store = ExampleStore(db_path=db_path, table_name="travel")
    ensure_seeded(store)

    hotel_agent = ExampleSelectorAgent(
        name="hotel_specialist",
        model=MODEL,
        instruction=HOTEL_INSTRUCTION,
        example_store=store,
        example_category="hotels",
        example_k=3,
    )

    flight_agent = ExampleSelectorAgent(
        name="flight_specialist",
        model=MODEL,
        instruction=FLIGHT_INSTRUCTION,
        example_store=store,
        example_category="flights",
        example_k=3,
    )

    router = LlmAgent(
        name="travel_router",
        model=MODEL,
        instruction=ROUTER_INSTRUCTION,
        sub_agents=[hotel_agent, flight_agent],
    )

    return TravelApp(
        router=router,
        hotel_agent=hotel_agent,
        flight_agent=flight_agent,
        store=store,
    )
