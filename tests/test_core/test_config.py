"""Tests for EmbeddingConfig."""

from example_selector.core.config import EmbeddingConfig


class TestEmbeddingConfig:
    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.model_type == "sentence-transformers"
        assert config.dimensions == 384

    def test_custom_values(self):
        config = EmbeddingConfig(
            model_name="custom-model",
            model_type="sentence-transformers",
            dimensions=768,
        )
        assert config.model_name == "custom-model"
        assert config.dimensions == 768

    def test_create_embedding_function(self):
        config = EmbeddingConfig()
        embed_fn = config.create_embedding_function()
        assert embed_fn is not None
        assert embed_fn.ndims() == 384

    def test_extra_kwargs(self):
        config = EmbeddingConfig(extra_kwargs={"device": "cpu"})
        assert config.extra_kwargs == {"device": "cpu"}
