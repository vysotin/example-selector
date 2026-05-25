# Travel Assistant Examples — Design Spec

**Date:** 2026-05-24
**Status:** Approved

## Goal

Add a worked example for each framework adapter (Google ADK, Langchain, Langgraph) that demonstrates the `example-selector` library on a realistic multi-agent task. Each demo is a travel-booking assistant with a router and two specialist subagents (hotels and flights). Each demo ships with a test script that validates correct example injection based on user intent, plus a comprehensive top-level `README.md`.

## Why

The repo currently has unit and functional tests but no end-to-end "look-what-you-can-build" demos. New users need:
1. A concrete, copy-pasteable starting point for each framework.
2. Proof that intent-based example injection works in a multi-agent setup (the most common production pattern).
3. A README that orients them quickly.

## Scope

In:
- New `examples/` directory tree with three self-contained demos and one shared dataset.
- Per-demo test script validating injection + live LLM round-trip.
- Top-level `README.md`.

Out of scope:
- No changes to the core framework (`example_selector/`). The demos consume existing public APIs only.
- No new injection mechanisms. Each demo uses the framework's **primary** mechanism only (per user clarification): `ExampleSelectorAgent` (ADK), `LanceDBExampleSelector` in `FewShotChatMessagePromptTemplate` (Langchain), `create_example_prompt` (Langgraph).

## Directory Layout

```
examples/
├── data/
│   └── travel_examples.json        # shared golden dataset (~20 examples)
├── adk_travel_assistant/
│   ├── README.md
│   ├── agent.py                    # builds router + 2 ExampleSelectorAgent specialists
│   ├── run.py                      # entrypoint: runs sample queries via ADK Runner
│   └── test_injection.py           # pytest: injection + live round-trip
├── langchain_travel_assistant/
│   ├── README.md
│   ├── agent.py                    # builds 2 specialist chains as Tools + router AgentExecutor
│   ├── run.py
│   └── test_injection.py
└── langgraph_travel_assistant/
    ├── README.md
    ├── agent.py                    # builds StateGraph: classifier → conditional → react agent
    ├── run.py
    └── test_injection.py
```

## Shared Dataset

`examples/data/travel_examples.json` — a JSON array of ~20 entries. Each entry:

```json
{
  "input_text": "I want to book a hotel in Paris for 3 nights",
  "output_text": "I'd be happy to help. To find the right hotel in Paris, could you share check-in/check-out dates, number of guests, and your preferred area or budget?",
  "category": "hotels",
  "metadata": ""
}
```

- ~10 `hotels` examples: search, book, cancel, modify, breakfast/amenities, room types, cancellation policy, group booking, loyalty points, special requests.
- ~10 `flights` examples: search, book, cancel, change date, baggage policy, seat selection, flight status, layover, refund, upgrade.

Loaded once at startup by each demo (via a small `_load_examples()` helper that resolves the JSON path relative to the demo file).

## ADK Demo

`agent.py` exports `build_app()` returning `(router_agent, store)`. Internals:

```python
store = ExampleStore(db_path="./examples_db", table_name="travel")
store.add_examples(load_travel_examples())  # idempotent — check store.count() first

hotel_agent = ExampleSelectorAgent(
    name="hotel_specialist",
    model="gemini-2.0-flash",
    instruction="You are a hotel booking specialist...",
    example_store=store,
    example_category="hotels",
    example_k=3,
)
flight_agent = ExampleSelectorAgent(
    name="flight_specialist",
    model="gemini-2.0-flash",
    instruction="You are a flight booking specialist...",
    example_store=store,
    example_category="flights",
    example_k=3,
)
router = LlmAgent(
    name="travel_router",
    model="gemini-2.0-flash",
    instruction="Route the user to the right specialist...",
    sub_agents=[hotel_agent, flight_agent],
)
```

`run.py`: uses ADK `InMemoryRunner` to send 2-3 sample queries (one hotel-y, one flight-y) and print the assistant response.

## Langchain Demo

Langchain has no native subagent abstraction (per user clarification). Each specialist is a chain wrapped as a `Tool`; the router is an `AgentExecutor`.

```python
hotel_selector = LanceDBExampleSelector(store=store, k=3, category="hotels")
hotel_few_shot = FewShotChatMessagePromptTemplate(
    example_selector=hotel_selector,
    example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
    input_variables=["input"],
)
hotel_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a hotel booking specialist..."),
    hotel_few_shot,
    ("human", "{input}"),
])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
hotel_chain = hotel_prompt | llm

hotel_tool = Tool.from_function(
    func=lambda q: hotel_chain.invoke({"input": q}).content,
    name="hotel_specialist",
    description="Use for hotel booking, cancellation, modification, room types, amenities.",
)
# (analogous for flight_tool)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel coordinator. Pick the right specialist tool for the user's request."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, [hotel_tool, flight_tool], router_prompt)
executor = AgentExecutor(agent=agent, tools=[hotel_tool, flight_tool])
```

`run.py`: invokes the executor with sample queries.

## Langgraph Demo

```python
hotel_prompt_fn = create_example_prompt(
    store, system_message="You are a hotel specialist...", k=3, category="hotels"
)
hotel_agent = create_react_agent(model, tools=[], prompt=hotel_prompt_fn)

flight_prompt_fn = create_example_prompt(
    store, system_message="You are a flight specialist...", k=3, category="flights"
)
flight_agent = create_react_agent(model, tools=[], prompt=flight_prompt_fn)

def classify(state: State) -> dict:
    last = state["messages"][-1].content
    decision = classifier_llm.invoke([
        SystemMessage("Classify: respond with exactly 'hotels' or 'flights'."),
        HumanMessage(last),
    ]).content.strip().lower()
    return {"intent": "hotels" if "hotel" in decision else "flights"}

builder = StateGraph(State)
builder.add_node("classifier", classify)
builder.add_node("hotel_specialist", hotel_agent)
builder.add_node("flight_specialist", flight_agent)
builder.add_edge(START, "classifier")
builder.add_conditional_edges("classifier", lambda s: s["intent"], {
    "hotels": "hotel_specialist", "flights": "flight_specialist"
})
builder.add_edge("hotel_specialist", END)
builder.add_edge("flight_specialist", END)
graph = builder.compile()
```

`State` is a `TypedDict` with `messages: Annotated[list, add_messages]` and `intent: str`.

## Test Strategy

Each demo's `test_injection.py` has two `pytest` classes:

### `TestInjection` (always runs, no API key)
- Calls each specialist's prompt/instruction assembly with a hotel query and a flight query.
- Asserts hotel examples appear in hotel agent's prompt and flight examples appear in flight agent's prompt.
- Per-framework implementation:
  - ADK: `await agent.canonical_instruction(ctx)` and inspect returned string.
  - Langchain: `chain.get_prompts()[0].format_messages(input=q)` and inspect messages.
  - Langgraph: call `prompt_fn(state)` directly and inspect returned messages.

### `TestLiveRoundTrip` (marked `@pytest.mark.live`, skipped without `GOOGLE_API_KEY`)
- Runs the full router/graph with Gemini against a hotel query.
- Asserts response text contains hotel-related keywords (and conversely for flight query).
- Single round-trip per intent to keep cost minimal.

Skip mechanism: top-of-file
```python
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
```
on the live class only — injection tests still run.

Register the `live` marker in `pyproject.toml` `[tool.pytest.ini_options]` to suppress warnings.

## Top-Level README.md

Structure:
1. **Overview** — one-paragraph elevator pitch.
2. **Why this exists** — the problem (manual few-shot wiring) and the solution (transparent injection).
3. **Installation** — `pip install -e ".[adk,langchain,langgraph]"`.
4. **Core concepts** — `ExampleStore`, `EmbeddingConfig`, similarity search, categories.
5. **Quick start per framework** — short code snippet for each.
6. **Worked example: travel booking assistant** — links to each demo folder.
7. **API reference** — table of public symbols per module.
8. **Configuration** — embedding model, db path, k, category filter.
9. **Running tests** — `pytest` for offline, `pytest -m live` for full.
10. **Development** — project layout.

## Key Decisions

| Question | Decision | Source |
|---|---|---|
| Real API or mock? | Real Gemini calls in demos and live tests | user |
| Subagent pattern | Router + 2 specialists | user |
| Test depth | Injection + live LLM round-trip | user |
| Langchain subagents | Specialists as Tools, AgentExecutor router | user |
| Dataset location | Shared `examples/data/travel_examples.json` | user |
| Injection coverage | Primary mechanism only per framework | user |
| Live test gating | `GOOGLE_API_KEY` env var, `@pytest.mark.live` | designer |
| Default model | `gemini-2.0-flash` | matches existing functional tests |
| Examples DB path | `./examples_db` per demo (gitignored) | designer |

## Acceptance

- All three demos run end-to-end with `GOOGLE_API_KEY` set.
- `pytest examples/` passes the injection tests offline.
- `pytest -m live examples/` passes the live round-trip tests with a key set.
- `README.md` covers all three frameworks with runnable snippets.
- No changes to `example_selector/` source.
