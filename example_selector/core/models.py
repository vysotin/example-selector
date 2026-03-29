"""Pydantic models for LanceDB schema."""

from __future__ import annotations

from typing import Optional

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


def create_example_model(embed_fn) -> type[LanceModel]:
    """Create a LanceDB model class with the given embedding function.

    We need a factory because the Vector dimension and embedding function
    are set at class definition time via field annotations.
    """
    dim = embed_fn.ndims()

    class PromptExample(LanceModel):
        """A prompt example stored in LanceDB with auto-embedding."""

        id: str
        input_text: str = embed_fn.SourceField()
        output_text: str
        category: str = ""
        metadata: str = ""
        vector: Vector(dim) = embed_fn.VectorField()  # type: ignore[valid-type]

    return PromptExample


# Default model using all-MiniLM-L6-v2
_default_embed_fn = (
    get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")
)
PromptExample = create_example_model(_default_embed_fn)
