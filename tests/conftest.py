"""Shared test fixtures."""

import tempfile

import pytest

from example_selector.core.store import ExampleStore


SAMPLE_EXAMPLES = [
    {
        "input_text": "How do I greet someone?",
        "output_text": "Hello! How can I help you today?",
        "category": "greeting",
    },
    {
        "input_text": "Say goodbye politely",
        "output_text": "Goodbye! Have a wonderful day!",
        "category": "greeting",
    },
    {
        "input_text": "Calculate the sum of 2 and 3",
        "output_text": "The sum of 2 and 3 is 5.",
        "category": "math",
    },
    {
        "input_text": "What is the capital of France?",
        "output_text": "The capital of France is Paris.",
        "category": "geography",
    },
    {
        "input_text": "Translate hello to Spanish",
        "output_text": "Hola",
        "category": "translation",
    },
]


@pytest.fixture
def tmp_db_path():
    """Provide a temporary directory for the LanceDB database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def store(tmp_db_path):
    """Provide an ExampleStore with sample data."""
    s = ExampleStore(db_path=tmp_db_path, table_name="test")
    s.add_examples(SAMPLE_EXAMPLES)
    return s


@pytest.fixture
def empty_store(tmp_db_path):
    """Provide an empty ExampleStore."""
    return ExampleStore(db_path=tmp_db_path, table_name="test_empty")
