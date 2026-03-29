"""Tests for ExampleStore."""

import tempfile

import pytest

from example_selector.core.store import ExampleStore
from example_selector.core.config import EmbeddingConfig


class TestExampleStoreInit:
    def test_creates_empty_store(self, empty_store):
        assert empty_store.count() == 0

    def test_custom_table_name(self, tmp_db_path):
        store = ExampleStore(db_path=tmp_db_path, table_name="custom")
        assert store.table_name == "custom"
        assert store.count() == 0

    def test_db_path_property(self, tmp_db_path):
        store = ExampleStore(db_path=tmp_db_path)
        assert store.db_path == tmp_db_path

    def test_embedding_config_property(self, tmp_db_path):
        config = EmbeddingConfig()
        store = ExampleStore(db_path=tmp_db_path, embedding_config=config)
        assert store.embedding_config is config

    def test_reopens_existing_table(self, tmp_db_path):
        store1 = ExampleStore(db_path=tmp_db_path, table_name="reopen")
        store1.add_example(input_text="test", output_text="result")

        store2 = ExampleStore(db_path=tmp_db_path, table_name="reopen")
        assert store2.count() == 1


class TestAddExample:
    def test_add_single(self, empty_store):
        eid = empty_store.add_example(
            input_text="hello", output_text="world"
        )
        assert isinstance(eid, str)
        assert len(eid) > 0
        assert empty_store.count() == 1

    def test_add_with_custom_id(self, empty_store):
        eid = empty_store.add_example(
            input_text="hello",
            output_text="world",
            example_id="my-id",
        )
        assert eid == "my-id"

    def test_add_with_category(self, empty_store):
        empty_store.add_example(
            input_text="hello",
            output_text="world",
            category="greet",
        )
        results = empty_store.list_examples(category="greet")
        assert len(results) == 1
        assert results[0]["category"] == "greet"

    def test_add_with_metadata(self, empty_store):
        empty_store.add_example(
            input_text="hello",
            output_text="world",
            metadata='{"source": "test"}',
        )
        results = empty_store.list_examples()
        assert results[0]["metadata"] == '{"source": "test"}'


class TestAddExamples:
    def test_add_multiple(self, empty_store):
        ids = empty_store.add_examples(
            [
                {"input_text": "a", "output_text": "1"},
                {"input_text": "b", "output_text": "2"},
                {"input_text": "c", "output_text": "3"},
            ]
        )
        assert len(ids) == 3
        assert empty_store.count() == 3

    def test_add_with_custom_ids(self, empty_store):
        ids = empty_store.add_examples(
            [
                {"id": "x1", "input_text": "a", "output_text": "1"},
                {"id": "x2", "input_text": "b", "output_text": "2"},
            ]
        )
        assert ids == ["x1", "x2"]

    def test_add_with_optional_fields(self, empty_store):
        empty_store.add_examples(
            [
                {
                    "input_text": "a",
                    "output_text": "1",
                    "category": "cat",
                    "metadata": "meta",
                },
            ]
        )
        results = empty_store.list_examples()
        assert results[0]["category"] == "cat"
        assert results[0]["metadata"] == "meta"


class TestSearch:
    def test_search_returns_results(self, store):
        results = store.search("hello greeting", k=2)
        assert len(results) <= 2
        assert all("input_text" in r for r in results)
        assert all("output_text" in r for r in results)
        assert all("_distance" in r for r in results)

    def test_search_no_vector_in_results(self, store):
        results = store.search("hello", k=1)
        assert all("vector" not in r for r in results)

    def test_search_respects_k(self, store):
        results = store.search("example", k=3)
        assert len(results) <= 3

    def test_search_with_category_filter(self, store):
        results = store.search("hello", k=10, category="greeting")
        assert all(r["category"] == "greeting" for r in results)

    def test_search_empty_store(self, empty_store):
        results = empty_store.search("hello", k=5)
        assert results == []

    def test_search_relevance_ordering(self, store):
        results = store.search("greet someone hello", k=5)
        # Results should be ordered by distance (ascending)
        distances = [r["_distance"] for r in results]
        assert distances == sorted(distances)


class TestListExamples:
    def test_list_all(self, store):
        results = store.list_examples()
        assert len(results) == 5

    def test_list_by_category(self, store):
        results = store.list_examples(category="greeting")
        assert len(results) == 2
        assert all(r["category"] == "greeting" for r in results)

    def test_list_no_vector(self, store):
        results = store.list_examples()
        assert all("vector" not in r for r in results)

    def test_list_empty_store(self, empty_store):
        results = empty_store.list_examples()
        assert results == []


class TestDelete:
    def test_delete_by_id(self, empty_store):
        eid = empty_store.add_example(input_text="x", output_text="y")
        assert empty_store.count() == 1
        empty_store.delete(eid)
        assert empty_store.count() == 0

    def test_delete_nonexistent_id(self, empty_store):
        # Should not raise
        empty_store.delete("nonexistent-id")


class TestClear:
    def test_clear(self, store):
        assert store.count() > 0
        store.clear()
        assert store.count() == 0

    def test_clear_empty_store(self, empty_store):
        empty_store.clear()
        assert empty_store.count() == 0


class TestCount:
    def test_count_empty(self, empty_store):
        assert empty_store.count() == 0

    def test_count_after_adds(self, empty_store):
        empty_store.add_example(input_text="a", output_text="1")
        assert empty_store.count() == 1
        empty_store.add_example(input_text="b", output_text="2")
        assert empty_store.count() == 2
