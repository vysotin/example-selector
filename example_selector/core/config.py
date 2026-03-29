"           ""Embedding configuration for LanceDB."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model used by LanceDB.

    Uses LanceDB's built-in embedding registry with sentence-transformers.
    """

    model_name: str = "all-MiniLM-L6-v2"
    model_type: str = "sentence-transformers"
    dimensions: int = 384
    extra_kwargs: dict = field(default_factory=dict)

    def create_embedding_function(self):
        """Create a LanceDB embedding function from the registry."""
        from lancedb.embeddings import get_registry

        return (
            get_registry()
            .get(self.model_type)
            .create(name=self.model_name, **self.extra_kwargs)
        )
