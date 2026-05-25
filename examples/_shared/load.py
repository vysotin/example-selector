"""Shared helpers for the travel-assistant example demos."""

from __future__ import annotations

import json
from pathlib import Path


_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "travel_examples.json"


def load_travel_examples() -> list[dict]:
    """Return the shared list of travel-assistant golden examples.

    Each entry has keys: input_text, output_text, category ('hotels' or 'flights'),
    metadata (free-text intent label).
    """
    with _DATA_PATH.open() as f:
        return json.load(f)


def ensure_seeded(store, examples: list[dict] | None = None) -> int:
    """Idempotently load the travel examples into ``store``.

    Returns the number of examples now in the store. Skips re-insertion if the
    store already has at least as many rows as the example set.
    """
    examples = examples if examples is not None else load_travel_examples()
    if store.count() >= len(examples):
        return store.count()
    store.add_examples(examples)
    return store.count()
