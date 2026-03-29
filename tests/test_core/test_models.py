"""Tests for core models."""

from example_selector.core.models import create_example_model, PromptExample
from example_selector.core.config import EmbeddingConfig


class TestCreateExampleModel:
    def test_creates_model_class(self):
        config = EmbeddingConfig()
        embed_fn = config.create_embedding_function()
        model_cls = create_example_model(embed_fn)
        assert model_cls is not None
        assert hasattr(model_cls, "model_fields")

    def test_model_has_expected_fields(self):
        config = EmbeddingConfig()
        embed_fn = config.create_embedding_function()
        model_cls = create_example_model(embed_fn)
        field_names = set(model_cls.model_fields.keys())
        assert "id" in field_names
        assert "input_text" in field_names
        assert "output_text" in field_names
        assert "category" in field_names
        assert "metadata" in field_names
        assert "vector" in field_names


class TestDefaultPromptExample:
    def test_default_model_exists(self):
        assert PromptExample is not None
        assert hasattr(PromptExample, "model_fields")
