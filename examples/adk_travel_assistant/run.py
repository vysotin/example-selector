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
