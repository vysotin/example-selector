"""LanceDB-backed example provider for Google ADK."""

from __future__ import annotations

from typing import Optional

from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example
from google.genai import types

from example_selector.core.store import ExampleStore


class LanceDBExampleProvider(BaseExampleProvider):
    """ADK BaseExampleProvider backed by a LanceDB ExampleStore.

    Retrieves semantically similar examples for a given query and converts
    them to ADK's Example format for use with ExampleTool.
    """

    def __init__(
        self,
        store: ExampleStore,
        k: int = 5,
        category: Optional[str] = None,
        input_role: str = "user",
        output_role: str = "model",
    ):
        self._store = store
        self._k = k
        self._category = category
        self._input_role = input_role
        self._output_role = output_role

    def get_examples(self, query: str) -> list[Example]:
        """Retrieve relevant examples from LanceDB by semantic similarity."""
        results = self._store.search(query, k=self._k, category=self._category)
        return [self._to_adk_example(r) for r in results]

    def _to_adk_example(self, record: dict) -> Example:
        """Convert a store record to an ADK Example."""
        return Example(
            input=types.Content(
                parts=[types.Part(text=record["input_text"])],
                role=self._input_role,
            ),
            output=[
                types.Content(
                    parts=[types.Part(text=record["output_text"])],
                    role=self._output_role,
                )
            ],
        )

    @property
    def store(self) -> ExampleStore:
        return self._store

    @property
    def k(self) -> int:
        return self._k

    @property
    def category(self) -> Optional[str]:
        return self._category
