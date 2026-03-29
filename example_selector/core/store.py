"""ExampleStore — LanceDB-backed vector store for prompt examples."""

from __future__ import annotations

import uuid
from typing import Optional

import lancedb

from example_selector.core.config import EmbeddingConfig
from example_selector.core.models import create_example_model


class ExampleStore:
    """Store and retrieve prompt examples using LanceDB with auto-embedding.

    Examples are embedded on insert using sentence-transformers and retrieved
    via similarity search for dynamic few-shot prompt construction.
    """

    def __init__(
        self,
        db_path: str = "./examples_db",
        table_name: str = "examples",
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self._db_path = db_path
        self._table_name = table_name
        self._config = embedding_config or EmbeddingConfig()
        self._embed_fn = self._config.create_embedding_function()
        self._model_class = create_example_model(self._embed_fn)
        self._db = lancedb.connect(db_path)
        self._table = None
        self._ensure_table()

    def _ensure_table(self):
        """Open or create the examples table."""
        tables = self._db.list_tables().tables
        if self._table_name in tables:
            self._table = self._db.open_table(self._table_name)
        else:
            self._table = self._db.create_table(
                self._table_name, schema=self._model_class
            )

    def add_example(
        self,
        input_text: str,
        output_text: str,
        category: str = "",
        metadata: str = "",
        example_id: Optional[str] = None,
    ) -> str:
        """Add a single example. Returns the example ID."""
        eid = example_id or str(uuid.uuid4())
        # Pass dicts — LanceDB auto-embeds via SourceField/VectorField
        self._table.add(
            [
                {
                    "id": eid,
                    "input_text": input_text,
                    "output_text": output_text,
                    "category": category,
                    "metadata": metadata,
                }
            ]
        )
        return eid

    def add_examples(
        self, examples: list[dict], id_field: str = "id"
    ) -> list[str]:
        """Add multiple examples. Each dict must have 'input_text' and 'output_text'.

        Optional fields: 'category', 'metadata', 'id'.
        Returns list of IDs.
        """
        records = []
        ids = []
        for ex in examples:
            eid = ex.get(id_field, str(uuid.uuid4()))
            ids.append(eid)
            records.append(
                {
                    "id": eid,
                    "input_text": ex["input_text"],
                    "output_text": ex["output_text"],
                    "category": ex.get("category", ""),
                    "metadata": ex.get("metadata", ""),
                }
            )
        self._table.add(records)
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Search for similar examples by query text.

        Returns list of dicts with keys: id, input_text, output_text, category,
        metadata, _distance (lower = more similar).
        """
        search_query = self._table.search(query).limit(k)
        if category:
            search_query = search_query.where(
                f"category = '{category}'", prefilter=True
            )
        results = search_query.to_list()
        # Remove the vector column from results for cleanliness
        for r in results:
            r.pop("vector", None)
        return results

    def list_examples(
        self, category: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """List all examples, optionally filtered by category."""
        if category:
            results = (
                self._table.search()
                .where(f"category = '{category}'")
                .limit(limit)
                .to_list()
            )
        else:
            results = self._table.search().limit(limit).to_list()
        for r in results:
            r.pop("vector", None)
            r.pop("_distance", None)
        return results

    def delete(self, example_id: str) -> None:
        """Delete an example by ID."""
        self._table.delete(f"id = '{example_id}'")

    def count(self) -> int:
        """Return the number of examples in the store."""
        return self._table.count_rows()

    def clear(self) -> None:
        """Delete all examples from the store."""
        self._table.delete("id IS NOT NULL")

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self._config
