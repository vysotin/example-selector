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
