# Langchain Travel Booking Assistant

A demo of `example_selector` integrated with **Langchain**: two specialist LCEL
chains (hotel + flight) each use `LanceDBExampleSelector` inside a
`FewShotChatMessagePromptTemplate`. Each chain is wrapped as a `Tool`, and a
router built with `langchain.agents.create_agent` (the LangChain 1.x
replacement for the legacy `AgentExecutor`) decides which specialist to call.

## What it shows
- `LanceDBExampleSelector` plugged into Langchain's native
  `FewShotChatMessagePromptTemplate`
- Specialist chains exposed as `Tool`s for tool-calling agents
- Intent-routing via `create_agent`

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
- `agent.py` — specialist chains + Tool wrappers + create_agent router
- `run.py` — sample queries through the agent
- `test_injection.py` — injection and live round-trip tests
