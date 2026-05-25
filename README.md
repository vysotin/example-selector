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
