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
