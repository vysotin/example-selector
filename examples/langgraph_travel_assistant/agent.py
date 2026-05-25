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
