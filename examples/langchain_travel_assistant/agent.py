"""Langchain travel-booking assistant.

Two specialist LCEL chains (hotel + flight) each use ``LanceDBExampleSelector``
inside a ``FewShotChatMessagePromptTemplate``. Each chain is wrapped as a
``Tool``, and a router agent built with ``langchain.agents.create_agent``
decides which tool to invoke per user request.

Note: ``create_agent`` is the LangChain 1.x replacement for the legacy
``AgentExecutor`` + ``create_tool_calling_agent``. It returns a compiled
graph that is invoked with ``{"messages": [HumanMessage(...)]}``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
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
    "tool. Pass the user's full question to the chosen tool, then return its "
    "answer to the user verbatim."
)


@dataclass
class TravelApp:
    executor: object  # CompiledStateGraph from create_agent
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
    """Assemble specialist chains, wrap them as tools, and build the router agent."""
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

    executor = create_agent(
        model=llm,
        tools=[hotel_tool, flight_tool],
        system_prompt=ROUTER_SYSTEM,
    )

    return TravelApp(
        executor=executor,
        hotel_chain=hotel_chain,
        flight_chain=flight_chain,
        hotel_prompt=hotel_prompt,
        flight_prompt=flight_prompt,
        store=store,
    )
