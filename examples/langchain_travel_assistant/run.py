"""Run the Langchain travel-booking assistant end-to-end.

Usage:
    export GOOGLE_API_KEY=...
    python -m examples.langchain_travel_assistant.run
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

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
        result = app.executor.invoke({"messages": [HumanMessage(content=q)]})
        last = result["messages"][-1]
        print(f"<<< ASSISTANT: {last.content}")


if __name__ == "__main__":
    main()
