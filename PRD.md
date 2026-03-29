# Example Selector Mini-Framework — PRD

## Overview

A Python mini-framework that provides **automatic, transparent injection of relevant few-shot examples** into agent prompts across three agentic frameworks: **Google ADK**, **Langchain**, and **Langgraph**. It uses **LanceDB** (file-based vector DB) with **lightweight local embedding models** (sentence-transformers) to perform local RAG over curated golden example datasets.

The goal is **zero-friction integration**: the user extends an Agent (or Runner, or PromptTemplate) with one extra parameter and gets automatic example injection — no manual prompt engineering, no ExampleTool wiring, no callback boilerplate.

## Design Principles

1. **Automatic & transparent** — Examples are injected automatically based on semantic similarity to the user query. The developer does not manually select or format examples.
2. **Minimal modification** — Extend native framework classes so that the extended versions can be used as drop-in replacements anywhere the originals are accepted.
3. **Shared core** — One vector store + embedding engine shared across all adapters.
4. **Lightweight** — File-based DB, no servers, local embeddings.
5. **Composable** — The extended classes remain fully compatible with all native framework features (sub-agents, tools, callbacks, plugins, etc.).

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  User Code                        │
│  Uses ExampleAgent / ExampleSelector / etc.       │
└────────┬──────────────┬──────────────┬────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼─────┐
    │ ADK     │   │ Langchain │  │ Langgraph│
    │ Adapter │   │ Adapter   │  │ Adapter  │
    └────┬────┘   └─────┬─────┘  └────┬─────┘
         │              │              │
    ┌────▼──────────────▼──────────────▼────┐
    │         Shared Core Engine             │
    │  ┌─────────────┐  ┌────────────────┐  │
    │  │ ExampleStore │  │ EmbeddingModel │  │
    │  │  (LanceDB)   │  │ (sent-transf.) │  │
    │  └─────────────┘  └────────────────┘  │
    └───────────────────────────────────────┘
```

---

## Shared Core Components

### `ExampleStore`
- Wraps LanceDB for CRUD + similarity search
- Uses LanceDB's Pydantic model integration with `SourceField`/`VectorField` for auto-embedding
- Schema: `input_text` (embedded for search), `output_text`, `category` (optional filter), `metadata` (optional JSON)
- Operations: `add_example()`, `add_examples()`, `search()`, `list_examples()`, `delete()`, `count()`, `clear()`
- Configurable: DB path, table name, embedding model name, top_k

### `EmbeddingConfig`
- Configures the embedding model via LanceDB's built-in registry
- Default: `sentence-transformers` with `all-MiniLM-L6-v2` (384 dimensions)
- Supports any model in LanceDB's embedding registry

---

## Google ADK Adapter — Design Decision Analysis

### Why NOT ExampleTool

The current implementation uses `BaseExampleProvider` + `ExampleTool`. This was rejected because:

1. **ExampleTool is a Tool, not prompt enrichment.** It occupies a slot in the agent's `tools` list alongside actual functional tools. Conceptually, example injection is a prompt construction concern, not a tool. Mixing the two is misleading.
2. **ExampleTool formatting is opaque.** It calls `example_util.build_example_si()` which formats examples using ADK's internal XML-like `<EXAMPLES>` template. The user has no control over formatting, placement, or structure.
3. **It requires ExampleTool awareness.** The user must import and wire `ExampleTool`, `BaseExampleProvider`, and understand ADK's example pipeline. This is not "automatic injection."
4. **It does not compose naturally.** If the user already has tools, adding ExampleTool is an unrelated concern mixed in. If they use `before_model_callback` for something else, ExampleTool may conflict.

### Approaches Considered

We analyzed four approaches for ADK integration. Here is the analysis:

#### Approach A: Subclass `LlmAgent` → `ExampleSelectorAgent`

**Mechanism:** Override `canonical_instruction()` to append dynamically selected examples to the agent's instruction text.

**How ADK prompt construction works (deep dive):**
1. `LlmAgent._run_async_impl()` delegates to `self._llm_flow.run_async(ctx)`
2. The flow calls `_preprocess_async()` which runs a chain of `request_processors`
3. Processor #4 is `_InstructionsLlmRequestProcessor` which calls `_build_instructions()`
4. `_build_instructions()` calls `agent.canonical_instruction(ctx)` to get the instruction text
5. The result is appended to `llm_request.config.system_instruction` via `llm_request.append_instructions()`
6. Later, `_process_agent_tools()` runs tool-specific processors (including ExampleTool)
7. Finally, `before_model_callback` runs on the fully assembled request

**`canonical_instruction` is the ideal override point because:**
- It runs early in the pipeline (step 4), before tools and callbacks
- It receives `ReadonlyContext` with `user_content`, `state`, `session` — everything needed for similarity search
- The return value is directly used as the system instruction — full control over formatting
- It's a regular method (not `@final`), explicitly designed to be overridable
- The agent remains a proper `LlmAgent` — all native features work (sub-agents, tools, callbacks, plugins)

```python
class ExampleSelectorAgent(LlmAgent):
    """LlmAgent that automatically injects relevant examples into its instruction."""

    example_store: ExampleStore
    example_k: int = 5
    example_category: Optional[str] = None
    example_format: str = "default"  # or "xml", "markdown", custom callable

    async def canonical_instruction(self, ctx: ReadonlyContext) -> tuple[str, bool]:
        # Get the base instruction (handles str vs callable)
        base_instruction, bypass = await super().canonical_instruction(ctx)

        # Extract query from user content
        query = self._extract_query(ctx)
        if not query:
            return base_instruction, bypass

        # RAG: find relevant examples
        examples = self.example_store.search(query, k=self.example_k, category=self.example_category)
        if not examples:
            return base_instruction, bypass

        # Format and append examples to instruction
        formatted = self._format_examples(examples)
        return f"{base_instruction}\n\n{formatted}", bypass
```

**Pros:**
- Cleanest API: just swap `LlmAgent` → `ExampleSelectorAgent` and add `example_store=`
- The agent IS-A `LlmAgent` — works everywhere `LlmAgent` works (sub-agents, orchestrators, etc.)
- Examples are part of the system instruction — the most natural place for few-shot guidance
- Full control over example formatting
- User's existing `instruction`, `tools`, `callbacks`, `sub_agents` all work unchanged
- Works with both string and callable instructions (via `super()`)
- Pydantic-compatible: extra fields declared on the subclass are allowed despite `extra='forbid'`

**Cons:**
- Couples to one agent. If the user has 10 agents, they'd need to change each one (but: usually only 1-2 agents need examples, and having per-agent control is a feature, not a bug)
- `canonical_instruction` is documented as "only for use by Agent Development Kit" — it's an internal API. However, it's not `@final`, not private (`_`), and subclassing `LlmAgent` is an explicitly supported pattern.

#### Approach B: `before_model_callback` factory

**Mechanism:** Provide a function that returns a `before_model_callback` callable. The callback modifies `llm_request` to append examples.

```python
agent = LlmAgent(
    name="my_agent",
    instruction="Be helpful.",
    before_model_callback=create_example_callback(store, k=3),
)
```

**Pros:**
- No subclassing, works with vanilla LlmAgent
- `before_model_callback` is a first-class ADK extension point
- Full access to `LlmRequest` (can modify system_instruction, contents, etc.)
- Async-friendly

**Cons:**
- **Occupies the callback slot.** If the user already has a `before_model_callback`, they must manually compose the two. ADK does support a list of callbacks, but the API ergonomics are worse.
- **Runs late in the pipeline.** The callback fires after instruction processors, content processors, and tool processors have all run. By this point the system instruction is fully assembled. Appending examples works, but they end up at the very end of the system instruction rather than logically after the agent's own instruction.
- **Not transparent.** The user sees a callback function — it's not immediately clear what it does. Compare to `ExampleSelectorAgent(example_store=store)` which is self-documenting.

**Decision:** Keep as a **secondary/escape-hatch option** for users who cannot subclass (e.g., they need to use vanilla `LlmAgent` for compatibility reasons).

#### Approach C: Callable `instruction` (InstructionProvider)

**Mechanism:** Wrap the user's instruction in a callable that appends examples.

```python
def make_instruction_with_examples(base_instruction, store, k=3):
    async def _provider(ctx: ReadonlyContext) -> str:
        query = ctx.user_content.parts[0].text if ctx.user_content else ""
        examples = store.search(query, k=k)
        return f"{base_instruction}\n\n{format_examples(examples)}"
    return _provider

agent = LlmAgent(
    name="my_agent",
    instruction=make_instruction_with_examples("Be helpful.", store),
)
```

**Pros:**
- No subclassing
- Works with vanilla LlmAgent

**Cons:**
- **Replaces the instruction entirely.** The user's instruction is now a closure, not a string. They can't easily inspect or modify it later. If they pass a callable, our wrapper would need to compose callables, which gets messy.
- **Bypasses state injection.** When `instruction` is a callable, ADK skips `{variable_name}` placeholder resolution. If the user relies on state injection in their instruction, this approach silently breaks it.
- **Poor discoverability.** A wrapped callable doesn't signal "this agent has example injection." Compare to a dedicated class.

**Decision:** **Rejected.** The state injection bypass is a dealbreaker for composability.

#### Approach D: Custom `BasePlugin` (Runner-level)

**Mechanism:** A plugin that runs `before_model_callback` globally for all agents in the runner.

```python
class ExampleSelectorPlugin(BasePlugin):
    def __init__(self, store, k=5):
        super().__init__(name="example_selector")
        self.store = store
        self.k = k

    async def before_model_callback(self, *, callback_context, llm_request):
        query = callback_context.user_content.parts[0].text
        examples = self.store.search(query, k=self.k)
        llm_request.append_instructions([format_examples(examples)])
        return None

# Usage:
app = App(name="my_app", root_agent=agent, plugins=[ExampleSelectorPlugin(store)])
```

**Pros:**
- Applied globally — useful when all agents should get examples
- Clean separation: plugin is a cross-cutting concern, separate from agent definition
- Works with any agent type, no subclassing needed

**Cons:**
- **Too broad.** Applied to ALL agents in the runner/app, including sub-agents that may not need examples. There's no built-in way to target specific agents.
- **Global scope conflicts.** If different agents need different example stores or categories, a single plugin can't handle it cleanly without internal routing logic.
- **Plugin runs before agent callbacks** — this is fine for injection, but it means the user can't override the plugin's behavior from within an agent.

**Decision:** Offer as an **optional complement** for the use case of "all agents get the same example store." But the primary approach should be per-agent (Approach A).

### Final ADK Design

**Primary: `ExampleSelectorAgent(LlmAgent)`** — Approach A
- Drop-in replacement for `LlmAgent`
- Accepts `example_store`, `example_k`, `example_category`, `example_format` as constructor params
- Overrides `canonical_instruction()` to automatically inject examples
- Preserves all `LlmAgent` features: tools, callbacks, sub-agents, plugins, state injection

**Secondary: `create_example_callback()`** — Approach B
- Returns a `before_model_callback` for users who need vanilla `LlmAgent`
- Less ergonomic but more flexible for edge cases

**Optional: `ExampleSelectorPlugin(BasePlugin)`** — Approach D
- For apps where all agents share the same example store
- Configured at the App/Runner level

### Detailed Usage Examples

#### Example 1: Basic ExampleSelectorAgent (Primary Pattern)

```python
from example_selector import ExampleStore
from example_selector.adk import ExampleSelectorAgent

# Set up the example store (once, at app startup)
store = ExampleStore("./golden_examples")
store.add_examples([
    {
        "input_text": "I can't login to my account",
        "output_text": "I'm sorry you're having trouble. Let me help you reset your password...",
        "category": "account",
    },
    {
        "input_text": "How do I upgrade my plan?",
        "output_text": "To upgrade, go to Settings > Billing > Change Plan...",
        "category": "billing",
    },
])

# Create the agent — just like LlmAgent but with example_store
agent = ExampleSelectorAgent(
    name="support_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful customer support agent. Be empathetic and professional.",
    example_store=store,
    example_k=3,           # Top 3 most relevant examples
    example_category=None, # Search all categories (or "billing" to filter)
    # All normal LlmAgent features work:
    tools=[search_tool, ticket_tool],
    sub_agents=[escalation_agent],
    output_key="response",
)
```

**What happens at runtime:**
1. User sends "I forgot my password"
2. ADK calls `agent.canonical_instruction(ctx)`
3. Our override calls `super().canonical_instruction(ctx)` → gets "You are a helpful customer support agent..."
4. Extracts "I forgot my password" from `ctx.user_content`
5. Searches LanceDB → finds the "I can't login" example (semantically similar)
6. Returns the instruction + formatted examples as the system instruction
7. ADK continues normally — tools, callbacks, etc. all work as usual

The agent's system instruction sent to the LLM looks like:
```
You are a helpful customer support agent. Be empathetic and professional.

Here are relevant examples to guide your response style:

Example 1:
  User: I can't login to my account
  Assistant: I'm sorry you're having trouble. Let me help you reset your password...
```

#### Example 2: ExampleSelectorAgent with Callable Instruction

```python
# Works with callable instructions too — state injection still works
async def dynamic_instruction(ctx: ReadonlyContext) -> str:
    user_tier = ctx.state.get("user_tier", "free")
    return f"You are a support agent for {user_tier} tier users."

agent = ExampleSelectorAgent(
    name="support",
    model="gemini-2.0-flash",
    instruction=dynamic_instruction,  # Callable instruction
    example_store=store,
    example_k=2,
)
# Our override calls super() which calls the callable, then appends examples
```

#### Example 3: ExampleSelectorAgent in Multi-Agent Setup

```python
from google.adk.agents import LlmAgent

# Only the agents that need examples use ExampleSelectorAgent
billing_agent = ExampleSelectorAgent(
    name="billing",
    instruction="Handle billing inquiries.",
    example_store=store,
    example_category="billing",
    example_k=3,
)

# Other agents remain plain LlmAgent — no changes needed
routing_agent = LlmAgent(
    name="router",
    instruction="Route the user to the right specialist.",
    sub_agents=[billing_agent, account_agent, general_agent],
)
# billing_agent IS-A LlmAgent, so it works perfectly as a sub-agent
```

#### Example 4: Custom Example Formatting

```python
# Default format: "Example N:\n  User: ...\n  Assistant: ..."
# But the user can customize:

def markdown_format(examples: list[dict]) -> str:
    lines = ["## Relevant Examples\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}")
        lines.append(f"**Input:** {ex['input_text']}")
        lines.append(f"**Output:** {ex['output_text']}")
        lines.append("")
    return "\n".join(lines)

agent = ExampleSelectorAgent(
    name="agent",
    instruction="Be helpful.",
    example_store=store,
    example_format=markdown_format,  # Custom formatter
)
```

#### Example 5: Plugin for Global Example Injection

```python
from example_selector.adk import ExampleSelectorPlugin
from google.adk.apps import App

# When ALL agents in the app should get examples from the same store
plugin = ExampleSelectorPlugin(store=store, k=3)

app = App(
    name="my_app",
    root_agent=router_agent,  # Can be plain LlmAgent
    plugins=[plugin],
)
```

---

## Langchain Adapter

**Approach**: Implement `BaseExampleSelector` (Langchain's native interface) for use with `FewShotChatMessagePromptTemplate`.

### Why `BaseExampleSelector`

Langchain already has a well-established pattern for few-shot example selection:
- `BaseExampleSelector` is an ABC with `add_example()` and `select_examples()` methods
- `FewShotChatMessagePromptTemplate` accepts an `example_selector` parameter
- The template automatically calls `select_examples()` with the input variables during formatting
- This composes naturally into `ChatPromptTemplate.from_messages()`

This is the only right approach — Langchain's prompt template system is designed for this exact use case.

### Components
- `LanceDBExampleSelector(BaseExampleSelector)` — implements:
  - `add_example(example: dict) -> str`
  - `select_examples(input_variables: dict) -> list[dict]`
  - Class method `from_examples(examples, db_path, **kwargs)`
- Configurable `input_key`, `output_key`, `example_keys` filter, `category` filter

### Usage

```python
from example_selector import ExampleStore
from example_selector.langchain import LanceDBExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

store = ExampleStore("./examples_db")
selector = LanceDBExampleSelector(store=store, k=3)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"), ("ai", "{output}")
])
few_shot = FewShotChatMessagePromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    input_variables=["input"],
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."), few_shot, ("human", "{input}")
])
# Use as: chain = prompt | llm
```

---

## Langgraph Adapter

### Design Decision

Langgraph does not have a built-in "example selector" pattern. The framework is graph-based: nodes are functions that transform state. We provide two integration points:

1. **`create_example_prompt(store, system_message, k)`** — Returns a callable suitable for `create_react_agent(prompt=...)`. This is the cleanest integration for the prebuilt react agent.

2. **`create_example_node(store, k)`** — Returns a node function for custom graphs. The node reads the last human message from state, finds similar examples, and prepends example messages to the state.

### Why Two Approaches

- `create_react_agent` accepts a `prompt` parameter that can be a callable `(state) -> list[BaseMessage]`. This is the perfect hook for injecting examples into the react agent's prompt without modifying the agent's internal logic.
- Custom Langgraph graphs need a node-based approach. The node can be inserted before the LLM node in any graph topology.

### Usage

```python
# With create_react_agent
from example_selector.langgraph import create_example_prompt
prompt = create_example_prompt(store, system_message="You are helpful.", k=3)
agent = create_react_agent(model, tools, prompt=prompt)

# With custom graph
from example_selector.langgraph import create_example_node
builder.add_node("inject_examples", create_example_node(store, k=3))
builder.add_edge(START, "inject_examples")
builder.add_edge("inject_examples", "agent")
```

---

## Package Structure

```
example_selector/
├── __init__.py              # ExampleStore, EmbeddingConfig exports
├── core/
│   ├── __init__.py
│   ├── store.py             # ExampleStore class
│   ├── models.py            # Pydantic models (PromptExample schema)
│   └── config.py            # EmbeddingConfig
├── adk/
│   ├── __init__.py
│   ├── agent.py             # ExampleSelectorAgent (PRIMARY)
│   ├── plugin.py            # ExampleSelectorPlugin (OPTIONAL)
│   └── helpers.py           # create_example_callback (SECONDARY)
├── langchain/
│   ├── __init__.py
│   └── selector.py          # LanceDBExampleSelector
└── langgraph/
    ├── __init__.py
    └── helpers.py            # create_example_node, create_example_prompt
```

---

## Dependencies

### Core
- `lancedb>=0.20` — file-based vector DB
- `sentence-transformers>=2.0` — local embedding models
- `pydantic>=2.0` — data models

### Framework-specific (optional)
- `google-adk>=1.0` — for ADK adapter
- `langchain-core>=0.3` — for Langchain adapter
- `langgraph>=0.2` — for Langgraph adapter

### Testing
- `pytest>=8.0`
- `pytest-asyncio>=0.23`

---

## Completion Criteria

1. **Unit test coverage**: All public methods of all classes have unit tests
2. **Functional tests**: End-to-end tests for each framework adapter demonstrating:
   - Adding examples to the store
   - Querying examples by similarity
   - Injecting examples into agent prompts via each framework's native pattern
   - Filtering examples by category
   - Multiple examples with ranking by relevance
   - Edge cases: empty store, unicode, long queries, special characters
3. **All tests pass**: `pytest` runs clean with 0 failures
4. **Code quality**: Type hints, docstrings on public API

## Implementation Order

1. Core engine (ExampleStore, models, config) — DONE
2. Google ADK adapter (ExampleSelectorAgent, plugin, callback helper) + tests
3. Langchain adapter + tests — DONE
4. Langgraph adapter + tests — DONE
