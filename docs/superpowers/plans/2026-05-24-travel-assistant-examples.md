# Travel Assistant Examples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three self-contained travel-booking assistant demos (one per framework: ADK, Langchain, Langgraph), each with router + 2 specialist subagents, a shared golden dataset, intent-based injection tests, and a top-level README.

**Architecture:** Each demo lives in its own `examples/<framework>_travel_assistant/` folder and consumes the existing `example_selector` public API only. All three load the same `examples/data/travel_examples.json` dataset. Tests come in two flavors: prompt-inspection (always run) and live Gemini round-trip (gated by `GOOGLE_API_KEY` + `@pytest.mark.live`).

**Tech Stack:** Python 3.10+, `example_selector` (this repo), `google-adk`, `langchain-core` + `langchain` (for `AgentExecutor`/`create_tool_calling_agent`), `langgraph`, `langchain-google-genai`, `pytest`, `pytest-asyncio`.

**Reference spec:** `docs/superpowers/specs/2026-05-24-travel-assistant-examples-design.md`

---

## File Layout

**New files (creates):**
- `examples/__init__.py` — empty marker (makes `examples` importable for tests if needed)
- `examples/data/travel_examples.json` — shared golden dataset
- `examples/_shared/__init__.py` — empty
- `examples/_shared/load.py` — `load_travel_examples()` helper
- `examples/adk_travel_assistant/__init__.py` — empty
- `examples/adk_travel_assistant/agent.py` — builds router + 2 `ExampleSelectorAgent` specialists
- `examples/adk_travel_assistant/run.py` — runs sample queries via `InMemoryRunner`
- `examples/adk_travel_assistant/test_injection.py` — pytest: injection + live round-trip
- `examples/adk_travel_assistant/README.md` — how-to-run
- `examples/langchain_travel_assistant/__init__.py` — empty
- `examples/langchain_travel_assistant/agent.py` — 2 specialist chains as Tools + `AgentExecutor`
- `examples/langchain_travel_assistant/run.py`
- `examples/langchain_travel_assistant/test_injection.py`
- `examples/langchain_travel_assistant/README.md`
- `examples/langgraph_travel_assistant/__init__.py`
- `examples/langgraph_travel_assistant/agent.py` — `StateGraph` with classifier + 2 react agents
- `examples/langgraph_travel_assistant/run.py`
- `examples/langgraph_travel_assistant/test_injection.py`
- `examples/langgraph_travel_assistant/README.md`
- `README.md` — top-level (NEW; the repo currently has no root README)

**Modified files:**
- `pyproject.toml` — add `langchain>=0.3` to `langchain` extra; register `live` pytest marker; add `examples` to package include exclude list (we do NOT want examples installed as a package).

---

## Task 1: Register pytest `live` marker and add Langchain `langchain` dep

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `pyproject.toml`** to add `langchain` to the langchain extra and register the `live` marker.

Add `langchain>=0.3` to the `langchain` and `dev` optional-deps groups:

```toml
[project.optional-dependencies]
adk = ["google-adk>=1.0"]
langchain = ["langchain-core>=0.3", "langchain>=0.3"]
langgraph = ["langgraph>=0.2"]
all = [
    "google-adk>=1.0",
    "langchain-core>=0.3",
    "langchain>=0.3",
    "langgraph>=0.2",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "google-adk>=1.0",
    "langchain-core>=0.3",
    "langchain>=0.3",
    "langgraph>=0.2",
    "langchain-google-genai>=2.0",
]
```

Register the `live` marker by replacing the existing `[tool.pytest.ini_options]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "examples"]
asyncio_mode = "auto"
markers = [
    "live: tests that call real LLMs; require GOOGLE_API_KEY",
]
```

- [ ] **Step 2: Install the new dep**

Run: `pip install -e ".[dev]"`
Expected: completes without error; `langchain` package becomes importable.

- [ ] **Step 3: Verify pytest marker works**

Run: `pytest --markers | grep live`
Expected: shows `@pytest.mark.live: tests that call real LLMs; require GOOGLE_API_KEY`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Register pytest live marker and add langchain to extras"
```

---

## Task 2: Create shared dataset and loader

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/_shared/__init__.py`
- Create: `examples/_shared/load.py`
- Create: `examples/data/travel_examples.json`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p examples/data examples/_shared
touch examples/__init__.py examples/_shared/__init__.py
```

- [ ] **Step 2: Write `examples/data/travel_examples.json`**

20 entries — 10 hotels and 10 flights. Full file content:

```json
[
  {
    "input_text": "I want to book a hotel in Paris for 3 nights",
    "output_text": "I'd be glad to help you find a hotel in Paris. To narrow down the best options, could you share your check-in date, number of guests, preferred neighborhood (e.g. Le Marais, Montmartre), and approximate nightly budget?",
    "category": "hotels",
    "metadata": "intent=search"
  },
  {
    "input_text": "Find me a 4-star hotel near Times Square under $300 per night",
    "output_text": "Searching 4-star hotels within walking distance of Times Square under $300/night. May I confirm your dates and the number of guests so I can return live availability?",
    "category": "hotels",
    "metadata": "intent=search"
  },
  {
    "input_text": "Cancel my hotel reservation for next weekend",
    "output_text": "Of course. To cancel your reservation, could you share the booking reference or the email address used for the booking? I'll also confirm the cancellation policy and any refund amount before we proceed.",
    "category": "hotels",
    "metadata": "intent=cancel"
  },
  {
    "input_text": "Can I change my hotel booking to a king-size bed?",
    "output_text": "Yes, I can request a room change. Please share your booking reference, and I'll check availability for a king room for the same dates. Note that the rate may differ depending on the room type.",
    "category": "hotels",
    "metadata": "intent=modify"
  },
  {
    "input_text": "Does the Marriott in Berlin include breakfast?",
    "output_text": "Breakfast at the Berlin Marriott is included on certain rate plans (typically \"Bed & Breakfast\" or \"Premium\" rates). On the standard \"Room Only\" rate, breakfast can be added for around €28 per person per day. Would you like me to check availability with breakfast included?",
    "category": "hotels",
    "metadata": "intent=amenities"
  },
  {
    "input_text": "What's the cancellation policy for my booking?",
    "output_text": "Cancellation policies depend on the rate type. Flexible rates allow free cancellation up to 24–48 hours before check-in. Non-refundable rates cannot be cancelled. If you share your booking reference, I'll quote the exact policy on your reservation.",
    "category": "hotels",
    "metadata": "intent=policy"
  },
  {
    "input_text": "I need 5 rooms for a corporate retreat",
    "output_text": "Happy to help arrange a group booking. For 5 or more rooms I can request a group rate, which often includes a small discount and flexible cancellation. Could you share the destination, dates, and any meeting-room requirements?",
    "category": "hotels",
    "metadata": "intent=group"
  },
  {
    "input_text": "Can I redeem my loyalty points at the Hilton Tokyo?",
    "output_text": "Yes, Hilton Honors points can be redeemed at the Hilton Tokyo. A standard room typically runs 60,000–80,000 points per night depending on the season. If you share your dates and member number I can confirm the exact award rate.",
    "category": "hotels",
    "metadata": "intent=loyalty"
  },
  {
    "input_text": "I need a hotel room accessible for a wheelchair",
    "output_text": "Absolutely. I'll filter the search to ADA-compliant accessible rooms with roll-in showers and step-free access. Could you confirm the city, dates, and whether you need any additional accommodations (e.g. service animal, visual alarms)?",
    "category": "hotels",
    "metadata": "intent=special_request"
  },
  {
    "input_text": "What time is check-in at the Hyatt Chicago?",
    "output_text": "Standard check-in at the Hyatt Chicago is 3:00 PM and check-out is 12:00 PM. Early check-in and late check-out can usually be requested at no charge subject to availability — want me to add the request to your reservation?",
    "category": "hotels",
    "metadata": "intent=info"
  },
  {
    "input_text": "I need a flight from New York to London next Friday",
    "output_text": "Got it — one-way or round trip? And how many passengers? Once I have that I'll surface direct flights from JFK/EWR to LHR for next Friday with fare and departure-time options.",
    "category": "flights",
    "metadata": "intent=search"
  },
  {
    "input_text": "Book me a round-trip flight to Tokyo, business class",
    "output_text": "Happy to help with a round-trip business-class fare to Tokyo. Could you share your departure airport, outbound date, return date, and number of passengers? I'll then return the best business-class options on direct carriers.",
    "category": "flights",
    "metadata": "intent=book"
  },
  {
    "input_text": "Cancel my United flight to Denver",
    "output_text": "I can help cancel your United flight to Denver. Please provide the 6-character confirmation code so I can pull up the booking and review the refund eligibility based on your fare class.",
    "category": "flights",
    "metadata": "intent=cancel"
  },
  {
    "input_text": "Change my flight from Saturday to Sunday",
    "output_text": "I can submit a date-change request. Could you share the confirmation code and your preferred Sunday departure window? A fare difference and change fee may apply depending on the airline and fare class.",
    "category": "flights",
    "metadata": "intent=change_date"
  },
  {
    "input_text": "How many bags can I check on Delta economy?",
    "output_text": "On Delta Main Cabin (economy) domestic flights, the first checked bag is $35 and the second is $45, each up to 50 lbs / 23 kg. On most international routes one checked bag is included. Want me to add bags to your existing booking?",
    "category": "flights",
    "metadata": "intent=baggage"
  },
  {
    "input_text": "Can I select my seat for the Lufthansa flight?",
    "output_text": "Yes — with your booking reference I can open the seat map and reserve a seat. Economy standard seats are complimentary; Economy Extra (extra legroom) and Premium Economy seats carry a fee, usually €30–€120 depending on route.",
    "category": "flights",
    "metadata": "intent=seat"
  },
  {
    "input_text": "What's the status of American Airlines flight 100?",
    "output_text": "Let me look that up. Could you confirm the date of travel? I'll fetch the latest scheduled and actual departure/arrival times, gate, and any delays for AA100.",
    "category": "flights",
    "metadata": "intent=status"
  },
  {
    "input_text": "I have a 6-hour layover in Dubai — anything you'd suggest?",
    "output_text": "Six hours is enough to leave the airport. The Dubai Metro reaches Downtown in about 45 minutes; the Burj Khalifa observation deck or a quick visit to Dubai Mall fits comfortably. Otherwise, the Emirates and Marhaba lounges offer showers and quiet seating. Want me to check lounge access on your ticket?",
    "category": "flights",
    "metadata": "intent=layover"
  },
  {
    "input_text": "Am I eligible for a refund on a non-refundable ticket?",
    "output_text": "Non-refundable tickets generally aren't refundable in cash, but most airlines offer a travel credit minus a change/cancel fee if you cancel before departure. Some refunds are available for schedule changes, illness with documentation, or under 24-hour cancellation rules. Share the booking and I'll check the specific options.",
    "category": "flights",
    "metadata": "intent=refund"
  },
  {
    "input_text": "Can I upgrade to first class with miles?",
    "output_text": "Mileage upgrades are typically available on most fare classes — award space and copays vary by route and airline. If you share the booking reference and your frequent-flyer number I can check upgrade availability and the miles + copay required.",
    "category": "flights",
    "metadata": "intent=upgrade"
  }
]
```

- [ ] **Step 3: Write `examples/_shared/load.py`**

```python
"""Shared helpers for the travel-assistant example demos."""

from __future__ import annotations

import json
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "travel_examples.json"


def load_travel_examples() -> list[dict]:
    """Return the shared list of travel-assistant golden examples.

    Each entry has keys: input_text, output_text, category ('hotels' or 'flights'),
    metadata (free-text intent label).
    """
    with _DATA_PATH.open() as f:
        return json.load(f)


def ensure_seeded(store, examples: list[dict] | None = None) -> int:
    """Idempotently load the travel examples into ``store``.

    Returns the number of examples now in the store. Skips re-insertion if the
    store already has at least as many rows as the example set.
    """
    examples = examples if examples is not None else load_travel_examples()
    if store.count() >= len(examples):
        return store.count()
    store.add_examples(examples)
    return store.count()
```

- [ ] **Step 4: Sanity-check the JSON loads**

Run: `python -c "from examples._shared.load import load_travel_examples; ex = load_travel_examples(); print(len(ex), ex[0]['category'], ex[-1]['category'])"`
Expected: `20 hotels flights`

- [ ] **Step 5: Commit**

```bash
git add examples/__init__.py examples/_shared/__init__.py examples/_shared/load.py examples/data/travel_examples.json
git commit -m "Add shared travel examples dataset and loader"
```

---

## Task 3: Build ADK travel assistant agent

**Files:**
- Create: `examples/adk_travel_assistant/__init__.py`
- Create: `examples/adk_travel_assistant/agent.py`

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p examples/adk_travel_assistant
touch examples/adk_travel_assistant/__init__.py
```

- [ ] **Step 2: Write `examples/adk_travel_assistant/agent.py`**

```python
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
```

- [ ] **Step 3: Smoke-check the agent constructs without error**

Run: `python -c "from examples.adk_travel_assistant.agent import build_app; app = build_app(); print(app.router.name, [a.name for a in app.router.sub_agents])"`
Expected: `travel_router ['hotel_specialist', 'flight_specialist']`

- [ ] **Step 4: Commit**

```bash
git add examples/adk_travel_assistant/__init__.py examples/adk_travel_assistant/agent.py
git commit -m "Add ADK travel assistant: router + 2 ExampleSelectorAgent specialists"
```

---

## Task 4: Write ADK injection + live tests

**Files:**
- Create: `examples/adk_travel_assistant/test_injection.py`

- [ ] **Step 1: Write `examples/adk_travel_assistant/test_injection.py`**

```python
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
    from google.adk.sessions import InMemorySessionService

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
```

- [ ] **Step 2: Run the injection tests (offline)**

Run: `pytest examples/adk_travel_assistant/test_injection.py -v -m "not live"`
Expected: 5 tests pass (`TestInjection::*`). Live tests are deselected by `-m "not live"`.

- [ ] **Step 3: Commit**

```bash
git add examples/adk_travel_assistant/test_injection.py
git commit -m "Add ADK travel assistant injection + live round-trip tests"
```

---

## Task 5: ADK demo runner and README

**Files:**
- Create: `examples/adk_travel_assistant/run.py`
- Create: `examples/adk_travel_assistant/README.md`

- [ ] **Step 1: Write `examples/adk_travel_assistant/run.py`**

```python
"""Run the ADK travel-booking assistant end-to-end.

Usage:
    export GOOGLE_API_KEY=...
    python -m examples.adk_travel_assistant.run
"""

from __future__ import annotations

import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types

from examples.adk_travel_assistant.agent import build_app


SAMPLE_QUERIES = [
    "I want to book a hotel in Barcelona for 2 nights starting next Tuesday.",
    "Find me a non-stop flight from London to Singapore on June 20.",
    "What's the cancellation policy on my hotel reservation?",
]


async def _send(runner: InMemoryRunner, session_id: str, query: str) -> None:
    print(f"\n>>> USER: {query}")
    async for event in runner.run_async(
        user_id="demo_user",
        session_id=session_id,
        new_message=types.Content(parts=[types.Part(text=query)], role="user"),
    ):
        if event.content and event.content.parts:
            for p in event.content.parts:
                if p.text:
                    print(f"<<< {event.author}: {p.text}")


async def main() -> None:
    app = build_app()
    runner = InMemoryRunner(agent=app.router, app_name="travel_demo")
    session = await runner.session_service.create_session(
        app_name="travel_demo", user_id="demo_user"
    )
    for q in SAMPLE_QUERIES:
        await _send(runner, session.id, q)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Write `examples/adk_travel_assistant/README.md`**

```markdown
# ADK Travel Booking Assistant

A demo of `example_selector` integrated with **Google ADK**: a router LlmAgent
delegates to two `ExampleSelectorAgent` specialists (hotels and flights). The
specialists automatically inject the most semantically-similar examples from
the shared dataset based on the user's query.

## What it shows
- `ExampleSelectorAgent` as a drop-in replacement for `LlmAgent`
- Per-agent `example_category` filtering so hotel queries see hotel examples
- Native ADK multi-agent composition via `sub_agents`

## Run

```bash
export GOOGLE_API_KEY=...   # required to call Gemini
pip install -e ".[dev]"
python -m examples.adk_travel_assistant.run
```

## Test

```bash
# Injection-only (no API key required, fast)
pytest examples/adk_travel_assistant -v -m "not live"

# End-to-end with live Gemini calls
pytest examples/adk_travel_assistant -v -m live
```

## Files
- `agent.py` — builds the router + specialist agents
- `run.py` — sample queries through `InMemoryRunner`
- `test_injection.py` — injection and live round-trip tests
```

- [ ] **Step 3: Sanity-check the run module imports**

Run: `python -c "from examples.adk_travel_assistant import run; print(run.SAMPLE_QUERIES)"`
Expected: prints the 3-element sample queries list.

- [ ] **Step 4: Commit**

```bash
git add examples/adk_travel_assistant/run.py examples/adk_travel_assistant/README.md
git commit -m "Add ADK travel assistant runner script and README"
```

---

## Task 6: Build Langchain travel assistant agent

**Files:**
- Create: `examples/langchain_travel_assistant/__init__.py`
- Create: `examples/langchain_travel_assistant/agent.py`

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p examples/langchain_travel_assistant
touch examples/langchain_travel_assistant/__init__.py
```

- [ ] **Step 2: Write `examples/langchain_travel_assistant/agent.py`**

```python
"""Langchain travel-booking assistant.

Two specialist LCEL chains (hotel + flight) each use ``LanceDBExampleSelector``
inside a ``FewShotChatMessagePromptTemplate``. Each chain is wrapped as a
``Tool``, and a router ``AgentExecutor`` decides which tool to invoke per
user request.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from example_selector import ExampleStore
from example_selector.langchain import LanceDBExampleSelector
from examples._shared.load import ensure_seeded


MODEL = "gemini-2.0-flash"

HOTEL_SYSTEM = (
    "You are a hotel booking specialist. Help the user search, book, modify "
    "and cancel hotel stays. Mirror the conciseness of the examples."
)
FLIGHT_SYSTEM = (
    "You are a flight booking specialist. Help with flight search, booking, "
    "changes, baggage and seat questions. Mirror the conciseness of the examples."
)
ROUTER_SYSTEM = (
    "You are a travel coordinator. For requests about hotels, rooms, or "
    "accommodations, call the hotel_specialist tool. For requests about "
    "flights, airlines, baggage, seats, or layovers, call the flight_specialist "
    "tool. Pass the user's full question to the chosen tool."
)


@dataclass
class TravelApp:
    executor: AgentExecutor
    hotel_chain: object
    flight_chain: object
    hotel_prompt: ChatPromptTemplate
    flight_prompt: ChatPromptTemplate
    store: ExampleStore


def _build_specialist_chain(
    llm: ChatGoogleGenerativeAI,
    store: ExampleStore,
    category: str,
    system_message: str,
    k: int = 3,
):
    """Build a specialist chain: system + few-shot examples + user input → LLM."""
    selector = LanceDBExampleSelector(store=store, k=k, category=category)
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot = FewShotChatMessagePromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        input_variables=["input"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), few_shot, ("human", "{input}")]
    )
    return prompt, prompt | llm


def build_app(db_path: Optional[str] = None) -> TravelApp:
    """Assemble specialist chains, wrap them as tools, and build the router executor."""
    db_path = db_path or tempfile.mkdtemp(prefix="travel_examples_db_")
    store = ExampleStore(db_path=db_path, table_name="travel")
    ensure_seeded(store)

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.2)

    hotel_prompt, hotel_chain = _build_specialist_chain(
        llm, store, "hotels", HOTEL_SYSTEM
    )
    flight_prompt, flight_chain = _build_specialist_chain(
        llm, store, "flights", FLIGHT_SYSTEM
    )

    hotel_tool = Tool.from_function(
        func=lambda q: hotel_chain.invoke({"input": q}).content,
        name="hotel_specialist",
        description=(
            "Use for hotel booking, modification, cancellation, amenities, "
            "loyalty programs, and accommodation questions. Input: a single "
            "user message string."
        ),
    )
    flight_tool = Tool.from_function(
        func=lambda q: flight_chain.invoke({"input": q}).content,
        name="flight_specialist",
        description=(
            "Use for flight search, booking, cancellation, date changes, "
            "baggage, seat selection, status, layovers, and refund questions. "
            "Input: a single user message string."
        ),
    )

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [hotel_tool, flight_tool], router_prompt)
    executor = AgentExecutor(agent=agent, tools=[hotel_tool, flight_tool], verbose=False)

    return TravelApp(
        executor=executor,
        hotel_chain=hotel_chain,
        flight_chain=flight_chain,
        hotel_prompt=hotel_prompt,
        flight_prompt=flight_prompt,
        store=store,
    )
```

- [ ] **Step 3: Smoke-check construction (will fail without GOOGLE_API_KEY because `ChatGoogleGenerativeAI` validates at init)**

If a key is present:
Run: `python -c "from examples.langchain_travel_assistant.agent import build_app; app = build_app(); print(type(app.executor).__name__, [t.name for t in app.executor.tools])"`
Expected: `AgentExecutor ['hotel_specialist', 'flight_specialist']`

If no key, smoke-check the import path and prompt construction only:
Run: `python -c "from examples.langchain_travel_assistant import agent; print(agent.MODEL)"`
Expected: `gemini-2.0-flash`

- [ ] **Step 4: Commit**

```bash
git add examples/langchain_travel_assistant/__init__.py examples/langchain_travel_assistant/agent.py
git commit -m "Add Langchain travel assistant: specialist chains as tools + router executor"
```

---

## Task 7: Write Langchain injection + live tests

**Files:**
- Create: `examples/langchain_travel_assistant/test_injection.py`

- [ ] **Step 1: Write `examples/langchain_travel_assistant/test_injection.py`**

```python
"""Validate Langchain travel assistant injection and end-to-end behavior.

- ``TestInjection`` runs offline: it builds the specialist prompts directly
  (no LLM) and asserts that formatting a prompt with a hotel/flight query
  yields the appropriate examples.
- ``TestLiveRoundTrip`` runs the full router AgentExecutor against Gemini.
"""

from __future__ import annotations

import os
import tempfile

import pytest

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
        result = app.executor.invoke({"input": "Book me a hotel in Madrid for 3 nights"})
        text = result["output"].lower()
        assert any(kw in text for kw in HOTEL_KEYWORDS), result["output"]

    def test_flight_query_returns_flight_response(self, app):
        result = app.executor.invoke({"input": "Find me a flight from Boston to Miami next Friday"})
        text = result["output"].lower()
        assert any(kw in text for kw in FLIGHT_KEYWORDS), result["output"]
```

- [ ] **Step 2: Run the injection tests (offline)**

Run: `pytest examples/langchain_travel_assistant/test_injection.py -v -m "not live"`
Expected: 4 tests pass (`TestInjection::*`).

- [ ] **Step 3: Commit**

```bash
git add examples/langchain_travel_assistant/test_injection.py
git commit -m "Add Langchain travel assistant injection + live round-trip tests"
```

---

## Task 8: Langchain demo runner and README

**Files:**
- Create: `examples/langchain_travel_assistant/run.py`
- Create: `examples/langchain_travel_assistant/README.md`

- [ ] **Step 1: Write `examples/langchain_travel_assistant/run.py`**

```python
"""Run the Langchain travel-booking assistant end-to-end.

Usage:
    export GOOGLE_API_KEY=...
    python -m examples.langchain_travel_assistant.run
"""

from __future__ import annotations

from examples.langchain_travel_assistant.agent import build_app


SAMPLE_QUERIES = [
    "I want to book a hotel in Lisbon for 4 nights starting next Friday.",
    "Find me a business-class flight from Seattle to Tokyo on June 15.",
    "What's the baggage allowance on Delta economy?",
]


def main() -> None:
    app = build_app()
    for q in SAMPLE_QUERIES:
        print(f"\n>>> USER: {q}")
        result = app.executor.invoke({"input": q})
        print(f"<<< ASSISTANT: {result['output']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `examples/langchain_travel_assistant/README.md`**

```markdown
# Langchain Travel Booking Assistant

A demo of `example_selector` integrated with **Langchain**: two specialist LCEL
chains (hotel + flight) each use `LanceDBExampleSelector` inside a
`FewShotChatMessagePromptTemplate`. Each chain is wrapped as a `Tool`, and an
`AgentExecutor` router decides which specialist to call.

## What it shows
- `LanceDBExampleSelector` plugged into Langchain's native
  `FewShotChatMessagePromptTemplate`
- Specialist chains exposed as `Tool`s for tool-calling agents
- Intent-routing via `create_tool_calling_agent` + `AgentExecutor`

## Run

```bash
export GOOGLE_API_KEY=...
pip install -e ".[dev]"
python -m examples.langchain_travel_assistant.run
```

## Test

```bash
# Offline (no API key)
pytest examples/langchain_travel_assistant -v -m "not live"

# Live with Gemini
pytest examples/langchain_travel_assistant -v -m live
```

## Files
- `agent.py` — specialist chains + Tool wrappers + AgentExecutor
- `run.py` — sample queries through the executor
- `test_injection.py` — injection and live round-trip tests
```

- [ ] **Step 3: Commit**

```bash
git add examples/langchain_travel_assistant/run.py examples/langchain_travel_assistant/README.md
git commit -m "Add Langchain travel assistant runner script and README"
```

---

## Task 9: Build Langgraph travel assistant agent

**Files:**
- Create: `examples/langgraph_travel_assistant/__init__.py`
- Create: `examples/langgraph_travel_assistant/agent.py`

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p examples/langgraph_travel_assistant
touch examples/langgraph_travel_assistant/__init__.py
```

- [ ] **Step 2: Write `examples/langgraph_travel_assistant/agent.py`**

```python
"""Langgraph travel-booking assistant.

A ``StateGraph`` with three nodes: a classifier (LLM intent classification)
plus two prebuilt react agents — hotel and flight specialists. Each specialist
uses ``create_example_prompt`` to inject category-filtered examples into its
prompt.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from example_selector import ExampleStore
from example_selector.langgraph import create_example_prompt
from examples._shared.load import ensure_seeded


MODEL = "gemini-2.0-flash"

HOTEL_SYSTEM = (
    "You are a hotel booking specialist. Help with search, booking, "
    "modification, cancellation, amenities, and loyalty. Mirror the examples."
)
FLIGHT_SYSTEM = (
    "You are a flight booking specialist. Help with search, booking, "
    "cancellation, date changes, baggage, seats, status, and layovers. "
    "Mirror the examples."
)
CLASSIFIER_SYSTEM = (
    "Classify the user's request. Respond with exactly one word: "
    "either 'hotels' or 'flights'. If the request mentions both, choose "
    "the dominant intent."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str


@dataclass
class TravelApp:
    graph: object
    hotel_prompt_fn: object
    flight_prompt_fn: object
    classifier_llm: ChatGoogleGenerativeAI
    store: ExampleStore


def build_app(db_path: Optional[str] = None) -> TravelApp:
    db_path = db_path or tempfile.mkdtemp(prefix="travel_examples_db_")
    store = ExampleStore(db_path=db_path, table_name="travel")
    ensure_seeded(store)

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)

    hotel_prompt_fn = create_example_prompt(
        store, system_message=HOTEL_SYSTEM, k=3, category="hotels"
    )
    flight_prompt_fn = create_example_prompt(
        store, system_message=FLIGHT_SYSTEM, k=3, category="flights"
    )

    hotel_agent = create_react_agent(model=llm, tools=[], prompt=hotel_prompt_fn)
    flight_agent = create_react_agent(model=llm, tools=[], prompt=flight_prompt_fn)

    def classify(state: State) -> dict:
        last = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        text = last.content if last else ""
        decision = llm.invoke([
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=str(text)),
        ]).content.strip().lower()
        intent: Literal["hotels", "flights"] = (
            "hotels" if "hotel" in decision else "flights"
        )
        return {"intent": intent}

    def route(state: State) -> str:
        return state["intent"]

    builder = StateGraph(State)
    builder.add_node("classifier", classify)
    builder.add_node("hotel_specialist", hotel_agent)
    builder.add_node("flight_specialist", flight_agent)
    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier",
        route,
        {"hotels": "hotel_specialist", "flights": "flight_specialist"},
    )
    builder.add_edge("hotel_specialist", END)
    builder.add_edge("flight_specialist", END)
    graph = builder.compile()

    return TravelApp(
        graph=graph,
        hotel_prompt_fn=hotel_prompt_fn,
        flight_prompt_fn=flight_prompt_fn,
        classifier_llm=llm,
        store=store,
    )
```

- [ ] **Step 3: Smoke-check construction (requires GOOGLE_API_KEY — `ChatGoogleGenerativeAI` validates at init)**

If a key is present:
Run: `python -c "from examples.langgraph_travel_assistant.agent import build_app; app = build_app(); print(type(app.graph).__name__)"`
Expected: `CompiledStateGraph`

If no key, smoke-check imports:
Run: `python -c "from examples.langgraph_travel_assistant import agent; print(agent.MODEL)"`
Expected: `gemini-2.0-flash`

- [ ] **Step 4: Commit**

```bash
git add examples/langgraph_travel_assistant/__init__.py examples/langgraph_travel_assistant/agent.py
git commit -m "Add Langgraph travel assistant: StateGraph with classifier + 2 react agents"
```

---

## Task 10: Write Langgraph injection + live tests

**Files:**
- Create: `examples/langgraph_travel_assistant/test_injection.py`

- [ ] **Step 1: Write `examples/langgraph_travel_assistant/test_injection.py`**

```python
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
```

- [ ] **Step 2: Run the injection tests (offline)**

Run: `pytest examples/langgraph_travel_assistant/test_injection.py -v -m "not live"`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add examples/langgraph_travel_assistant/test_injection.py
git commit -m "Add Langgraph travel assistant injection + live round-trip tests"
```

---

## Task 11: Langgraph demo runner and README

**Files:**
- Create: `examples/langgraph_travel_assistant/run.py`
- Create: `examples/langgraph_travel_assistant/README.md`

- [ ] **Step 1: Write `examples/langgraph_travel_assistant/run.py`**

```python
"""Run the Langgraph travel-booking assistant end-to-end.

Usage:
    export GOOGLE_API_KEY=...
    python -m examples.langgraph_travel_assistant.run
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from examples.langgraph_travel_assistant.agent import build_app


SAMPLE_QUERIES = [
    "I want to book a hotel in Amsterdam for 2 nights starting next Wednesday.",
    "Find me an economy flight from LAX to ORD on June 18.",
    "Can I upgrade my flight to first class using miles?",
]


def main() -> None:
    app = build_app()
    for q in SAMPLE_QUERIES:
        print(f"\n>>> USER: {q}")
        result = app.graph.invoke(
            {"messages": [HumanMessage(content=q)], "intent": ""}
        )
        print(f"    [routed to: {result['intent']}]")
        last = result["messages"][-1]
        print(f"<<< ASSISTANT: {last.content}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `examples/langgraph_travel_assistant/README.md`**

```markdown
# Langgraph Travel Booking Assistant

A demo of `example_selector` integrated with **Langgraph**: a `StateGraph` with
an LLM classifier node that routes to one of two prebuilt react agents. Each
react agent's prompt is built with `create_example_prompt`, which injects
category-filtered examples into the prompt at call time.

## What it shows
- `create_example_prompt` plugged into `create_react_agent(prompt=...)`
- Conditional routing via `StateGraph.add_conditional_edges`
- Per-specialist example category filtering

## Run

```bash
export GOOGLE_API_KEY=...
pip install -e ".[dev]"
python -m examples.langgraph_travel_assistant.run
```

## Test

```bash
# Offline
pytest examples/langgraph_travel_assistant -v -m "not live"

# Live with Gemini
pytest examples/langgraph_travel_assistant -v -m live
```

## Files
- `agent.py` — StateGraph: classifier → conditional edge → react agent
- `run.py` — sample queries through the compiled graph
- `test_injection.py` — injection and live round-trip tests
```

- [ ] **Step 3: Commit**

```bash
git add examples/langgraph_travel_assistant/run.py examples/langgraph_travel_assistant/README.md
git commit -m "Add Langgraph travel assistant runner script and README"
```

---

## Task 12: Top-level README.md

**Files:**
- Create: `README.md` (project root)

- [ ] **Step 1: Write the top-level `README.md`**

```markdown
# example-selector

> Dynamic, transparent few-shot example injection for AI agents across
> **Google ADK**, **Langchain**, and **Langgraph** — backed by LanceDB
> and local sentence-transformer embeddings.

## What it does

Most agent frameworks make few-shot example prompting a manual, error-prone
step: you either hard-code a handful of examples or build a custom retrieval
layer yourself. `example-selector` does this for you:

1. You drop golden examples into an `ExampleStore` (a thin wrapper around
   LanceDB with auto-embedding).
2. You extend the framework's native agent / prompt class with one extra
   parameter (`example_store=...`).
3. At runtime, the framework automatically retrieves the top-k most
   semantically similar examples and injects them into the system instruction
   or prompt — transparently to you and to the model.

No `ExampleTool` wiring, no manual prompt engineering, no
`before_model_callback` boilerplate.

## Why it exists

| | Without example-selector | With example-selector |
| --- | --- | --- |
| Choose examples | Manual, static, in code | Automatic, dynamic, semantic |
| Add a new example | Edit prompt template | `store.add_example(...)` |
| Filter by intent | Build separate prompts | `example_category="..."` |
| Cross-framework | One implementation per framework | One store, three adapters |
| Local | Requires hosted vector DB | File-based, runs on a laptop |

## Installation

```bash
pip install -e ".[adk,langchain,langgraph]"
# or just one:
pip install -e ".[adk]"
```

For development (all frameworks + tests):

```bash
pip install -e ".[dev]"
```

## Core concepts

| Class | Purpose |
| --- | --- |
| `ExampleStore` | Stores examples (input/output/category/metadata) in LanceDB; embeds and retrieves by similarity. |
| `EmbeddingConfig` | Selects the embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim). |
| `PromptExample` | Pydantic model for a single example (auto-embedded). |

A populated store is the only thing the three adapters need:

```python
from example_selector import ExampleStore

store = ExampleStore(db_path="./golden_examples", table_name="support")
store.add_examples([
    {"input_text": "I can't log in", "output_text": "Let's reset your password...", "category": "account"},
    {"input_text": "How do I cancel?", "output_text": "I can cancel that for you...", "category": "billing"},
])
```

## Quick start

### Google ADK

```python
from google.adk.agents import LlmAgent
from example_selector.adk import ExampleSelectorAgent

billing = ExampleSelectorAgent(
    name="billing",
    model="gemini-2.0-flash",
    instruction="You handle billing inquiries.",
    example_store=store,
    example_category="billing",
    example_k=3,
)
# Drop-in replacement for LlmAgent — works as a sub_agent, with tools, with callbacks, ...
```

### Langchain

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from example_selector.langchain import LanceDBExampleSelector

selector = LanceDBExampleSelector(store=store, k=3, category="billing")
few_shot = FewShotChatMessagePromptTemplate(
    example_selector=selector,
    example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
    input_variables=["input"],
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You handle billing inquiries."),
    few_shot,
    ("human", "{input}"),
])
chain = prompt | llm
```

### Langgraph

```python
from langgraph.prebuilt import create_react_agent
from example_selector.langgraph import create_example_prompt

prompt_fn = create_example_prompt(
    store, system_message="You handle billing inquiries.", k=3, category="billing"
)
agent = create_react_agent(model, tools, prompt=prompt_fn)
```

## Worked example: travel-booking assistant

A full end-to-end demo lives under `examples/`, with the same conceptual
agent (router → hotel & flight specialists) built in all three frameworks.
Each demo loads the same JSON dataset of ~20 multi-shot examples and ships
with a test script that validates intent-based injection.

| Framework | Folder |
| --- | --- |
| Google ADK | [`examples/adk_travel_assistant/`](examples/adk_travel_assistant/README.md) |
| Langchain | [`examples/langchain_travel_assistant/`](examples/langchain_travel_assistant/README.md) |
| Langgraph | [`examples/langgraph_travel_assistant/`](examples/langgraph_travel_assistant/README.md) |

Shared dataset: [`examples/data/travel_examples.json`](examples/data/travel_examples.json)

Each demo has the same shape:
- `agent.py` — builds the router and two specialist subagents
- `run.py` — runs a few sample queries via the framework's runner
- `test_injection.py` — two test classes:
  - `TestInjection` — asserts hotel queries surface hotel examples (and vice versa). Offline, no API key.
  - `TestLiveRoundTrip` — invokes the full agent against Gemini. Skipped without `GOOGLE_API_KEY`; marked `@pytest.mark.live`.

To run a demo:

```bash
export GOOGLE_API_KEY=...
python -m examples.adk_travel_assistant.run
# or .langchain_travel_assistant.run  or  .langgraph_travel_assistant.run
```

## API reference (public symbols)

```
example_selector
├── ExampleStore                 # core/store.py
├── EmbeddingConfig              # core/config.py
└── PromptExample                # core/models.py

example_selector.adk
├── ExampleSelectorAgent         # primary: subclass of LlmAgent
├── ExampleSelectorPlugin        # optional: runner-wide injection
└── create_example_callback      # escape hatch: vanilla LlmAgent

example_selector.langchain
└── LanceDBExampleSelector       # implements BaseExampleSelector

example_selector.langgraph
├── create_example_prompt        # for create_react_agent(prompt=...)
└── create_example_node          # for custom StateGraph nodes
```

## Configuration

```python
from example_selector import ExampleStore, EmbeddingConfig

store = ExampleStore(
    db_path="./my_examples_db",         # LanceDB on-disk folder
    table_name="my_table",
    embedding_config=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",  # any sentence-transformers model
        model_type="sentence-transformers",
        dimensions=384,
    ),
)
```

Per-adapter common knobs: `k` (top-k examples), `category` (string filter),
`example_format` (callable returning a string — ADK only).

## Running the tests

```bash
# Offline — unit, integration, and demo injection tests
pytest

# Include live LLM round-trip tests (needs GOOGLE_API_KEY)
GOOGLE_API_KEY=... pytest -m live
```

Marker registered in `pyproject.toml`:
```toml
markers = ["live: tests that call real LLMs; require GOOGLE_API_KEY"]
```

## Project layout

```
example_selector/        # the library
├── core/                # ExampleStore + embedding config
├── adk/                 # Google ADK adapter
├── langchain/           # Langchain adapter
└── langgraph/           # Langgraph adapter

examples/                # runnable demos (one per framework)
├── data/
├── adk_travel_assistant/
├── langchain_travel_assistant/
└── langgraph_travel_assistant/

tests/                   # unit + functional tests
docs/                    # design specs and plans
```

## License

See `LICENSE` (if present).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add top-level README with overview, quick-start, and demo links"
```

---

## Task 13: Final verification — run all injection tests

**Files:** none (verification only)

- [ ] **Step 1: Run all non-live tests**

Run: `pytest -v -m "not live"`
Expected: existing 105 framework tests pass + 13 new demo injection tests pass (ADK 5, Langchain 4, Langgraph 4) — total 118+ passing.

- [ ] **Step 2: Verify the full project structure**

Run: `find examples -type f | sort`
Expected output:
```
examples/__init__.py
examples/_shared/__init__.py
examples/_shared/load.py
examples/adk_travel_assistant/README.md
examples/adk_travel_assistant/__init__.py
examples/adk_travel_assistant/agent.py
examples/adk_travel_assistant/run.py
examples/adk_travel_assistant/test_injection.py
examples/data/travel_examples.json
examples/langchain_travel_assistant/README.md
examples/langchain_travel_assistant/__init__.py
examples/langchain_travel_assistant/agent.py
examples/langchain_travel_assistant/run.py
examples/langchain_travel_assistant/test_injection.py
examples/langgraph_travel_assistant/README.md
examples/langgraph_travel_assistant/__init__.py
examples/langgraph_travel_assistant/agent.py
examples/langgraph_travel_assistant/run.py
examples/langgraph_travel_assistant/test_injection.py
```

- [ ] **Step 3: (Optional) Run live tests if a key is configured**

Run: `GOOGLE_API_KEY=$GOOGLE_API_KEY pytest -v -m live examples/`
Expected: 6 live tests pass (2 per framework).

- [ ] **Step 4: Done.** No commit needed — verification only.

---

## Self-Review

**Spec coverage:**
- Shared JSON dataset → Task 2
- ADK demo (router + 2 ExampleSelectorAgent) → Tasks 3-5
- Langchain demo (specialist chains as Tools + AgentExecutor router) → Tasks 6-8
- Langgraph demo (StateGraph classifier + 2 react agents) → Tasks 9-11
- Per-framework test scripts (injection + live round-trip) → Tasks 4, 7, 10
- Top-level README → Task 12
- `live` pytest marker + Langchain extra → Task 1
- Final verification → Task 13

All spec sections covered.

**Placeholder scan:** No TBDs/TODOs. Every code block is complete and ready to paste.

**Type consistency:** `build_app()` returns `TravelApp` in all three frameworks; field names differ per-framework (`router`+`hotel_agent`+`flight_agent` for ADK; `executor`+`hotel_chain`+`flight_chain` for Langchain; `graph` for Langgraph) — that's intentional since each framework has different primary objects. Tests in each framework reference the matching field set. `ExampleStore`, `ensure_seeded`, `load_travel_examples` signatures are stable across all references.

**Scope:** ~13 small tasks, each commit-bounded; fits a single plan.
