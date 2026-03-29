"""LanceDB-backed ExampleSelector for Langchain."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.example_selectors.base import BaseExampleSelector

from example_selector.core.store import ExampleStore


class LanceDBExampleSelector(BaseExampleSelector):
    """Langchain ExampleSelector backed by a LanceDB ExampleStore.

    Selects examples by semantic similarity using LanceDB vector search.
    Works with FewShotPromptTemplate and FewShotChatMessagePromptTemplate.
    """

    def __init__(
        self,
        store: ExampleStore,
        k: int = 4,
        category: Optional[str] = None,
        input_key: str = "input",
        output_key: str = "output",
        example_keys: Optional[list[str]] = None,
    ):
        self._store = store
        self._k = k
        self._category = category
        self._input_key = input_key
        self._output_key = output_key
        self._example_keys = example_keys

    def add_example(self, example: dict[str, str]) -> Any:
        """Add an example to the store.

        The example dict should have the input_key and output_key fields.
        """
        input_text = example.get(self._input_key, "")
        output_text = example.get(self._output_key, "")
        category = example.get("category", "")
        return self._store.add_example(
            input_text=input_text,
            output_text=output_text,
            category=category,
        )

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select examples by similarity to the input variables.

        Uses the value of input_key from input_variables as the search query.
        Falls back to joining all input variable values if input_key is not present.
        """
        query = input_variables.get(self._input_key)
        if query is None:
            # Fall back: join all values
            query = " ".join(str(v) for v in input_variables.values())

        results = self._store.search(
            query=query, k=self._k, category=self._category
        )
        return [self._to_example_dict(r) for r in results]

    def _to_example_dict(self, record: dict) -> dict:
        """Convert a store record to the example dict format expected by Langchain."""
        example = {
            self._input_key: record["input_text"],
            self._output_key: record["output_text"],
        }
        if record.get("category"):
            example["category"] = record["category"]
        if record.get("metadata"):
            example["metadata"] = record["metadata"]

        if self._example_keys:
            example = {k: v for k, v in example.items() if k in self._example_keys}

        return example

    @classmethod
    def from_examples(
        cls,
        examples: list[dict],
        db_path: str = "./examples_db",
        table_name: str = "examples",
        k: int = 4,
        input_key: str = "input",
        output_key: str = "output",
        **kwargs,
    ) -> LanceDBExampleSelector:
        """Create a selector pre-populated with examples."""
        store = ExampleStore(db_path=db_path, table_name=table_name)
        selector = cls(
            store=store, k=k, input_key=input_key, output_key=output_key, **kwargs
        )
        for example in examples:
            selector.add_example(example)
        return selector

    @property
    def store(self) -> ExampleStore:
        return self._store

    @property
    def k(self) -> int:
        return self._k

    @property
    def category(self) -> Optional[str]:
        return self._category
