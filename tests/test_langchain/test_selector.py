"""Tests for LanceDBExampleSelector."""

import tempfile

import pytest

from langchain_core.example_selectors.base import BaseExampleSelector

from example_selector.langchain.selector import LanceDBExampleSelector


class TestLanceDBExampleSelector:
    def test_is_base_example_selector(self, store):
        selector = LanceDBExampleSelector(store=store)
        assert isinstance(selector, BaseExampleSelector)

    def test_select_examples_returns_list_of_dicts(self, store):
        selector = LanceDBExampleSelector(store=store, k=3)
        results = selector.select_examples({"input": "greet someone"})
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert len(results) <= 3

    def test_select_examples_has_input_output_keys(self, store):
        selector = LanceDBExampleSelector(store=store, k=1)
        results = selector.select_examples({"input": "hello"})
        assert len(results) >= 1
        assert "input" in results[0]
        assert "output" in results[0]

    def test_select_examples_custom_keys(self, store):
        selector = LanceDBExampleSelector(
            store=store, k=1, input_key="question", output_key="answer"
        )
        results = selector.select_examples({"question": "hello"})
        assert "question" in results[0]
        assert "answer" in results[0]

    def test_select_examples_category_filter(self, store):
        selector = LanceDBExampleSelector(
            store=store, k=10, category="greeting"
        )
        results = selector.select_examples({"input": "anything"})
        assert len(results) <= 2  # Only 2 greetings in sample

    def test_select_examples_joins_values_as_fallback(self, store):
        selector = LanceDBExampleSelector(store=store, k=2)
        # When input_key not in input_variables, joins all values
        results = selector.select_examples({"query": "hello greeting"})
        assert len(results) >= 1

    def test_add_example(self, empty_store):
        selector = LanceDBExampleSelector(store=empty_store, k=5)
        result = selector.add_example({"input": "test", "output": "result"})
        assert isinstance(result, str)
        assert empty_store.count() == 1

    def test_add_example_with_category(self, empty_store):
        selector = LanceDBExampleSelector(store=empty_store, k=5)
        selector.add_example(
            {"input": "test", "output": "result", "category": "test_cat"}
        )
        results = empty_store.list_examples(category="test_cat")
        assert len(results) == 1

    def test_example_keys_filter(self, store):
        selector = LanceDBExampleSelector(
            store=store, k=1, example_keys=["input", "output"]
        )
        results = selector.select_examples({"input": "hello"})
        assert set(results[0].keys()) == {"input", "output"}

    def test_empty_store(self, empty_store):
        selector = LanceDBExampleSelector(store=empty_store, k=5)
        results = selector.select_examples({"input": "hello"})
        assert results == []

    def test_properties(self, store):
        selector = LanceDBExampleSelector(
            store=store, k=3, category="math"
        )
        assert selector.store is store
        assert selector.k == 3
        assert selector.category == "math"


class TestFromExamples:
    def test_creates_selector_with_examples(self, tmp_db_path):
        examples = [
            {"input": "hello", "output": "Hi!"},
            {"input": "bye", "output": "Goodbye!"},
        ]
        selector = LanceDBExampleSelector.from_examples(
            examples, db_path=tmp_db_path, table_name="from_examples"
        )
        assert isinstance(selector, LanceDBExampleSelector)
        assert selector.store.count() == 2

    def test_from_examples_searchable(self, tmp_db_path):
        examples = [
            {"input": "calculate 2+2", "output": "4"},
            {"input": "greet the user", "output": "Hello!"},
        ]
        selector = LanceDBExampleSelector.from_examples(
            examples, db_path=tmp_db_path, table_name="searchable", k=1
        )
        results = selector.select_examples({"input": "math addition"})
        assert len(results) >= 1
