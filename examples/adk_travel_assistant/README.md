# ADK Travel Booking Assistant

A demo of `example_selector` integrated with **Google ADK**: a router LlmAgent
delegates to two `ExampleSelectorAgent` specialists (hotels and flights). The
specialists automatically inject the most semantically-similar examples from
the shared dataset based on the user's query.

## What it shows
- `ExampleSelectorAgent` as a drop-in replacement for `LlmAgent`
- Per-agent `example_category` filtering so hotel queries see hotel examples
- Native ADK multi-agent composition via `sub_agents`

## Run

```bash
export GOOGLE_API_KEY=...   # required to call Gemini
pip install -e ".[dev]"
python -m examples.adk_travel_assistant.run
```

## Test

```bash
# Injection-only (no API key required, fast)
pytest examples/adk_travel_assistant -v -m "not live"

# End-to-end with live Gemini calls
pytest examples/adk_travel_assistant -v -m live
```

## Files
- `agent.py` — builds the router + specialist agents
- `run.py` — sample queries through `InMemoryRunner`
- `test_injection.py` — injection and live round-trip tests
