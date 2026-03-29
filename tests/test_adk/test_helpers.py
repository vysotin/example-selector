"""Tests for ADK helper functions."""

from unittest.mock import MagicMock

import pytest

from google.genai import types

from example_selector.adk.helpers import create_example_callback


class TestCreateExampleCallback:
    def _make_context(self, text: str):
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(text=text)], role="user"
        )
        return ctx

    def _make_llm_request(self):
        req = MagicMock()
        req.append_instructions = MagicMock()
        return req

    def test_callback_appends_instructions(self, store):
        callback = create_example_callback(store, k=2)
        ctx = self._make_context("greet someone")
        req = self._make_llm_request()
        result = callback(ctx, req)
        assert result is None  # Should continue with LLM call
        req.append_instructions.assert_called_once()
        instruction = req.append_instructions.call_args[0][0][0]
        assert "Example" in instruction

    def test_callback_skips_empty_content(self, store):
        callback = create_example_callback(store)
        ctx = MagicMock()
        ctx.user_content = types.Content(parts=[], role="user")
        req = self._make_llm_request()
        result = callback(ctx, req)
        assert result is None
        req.append_instructions.assert_not_called()

    def test_callback_skips_no_text_part(self, store):
        callback = create_example_callback(store)
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(inline_data=types.Blob(data=b"img", mime_type="image/png"))],
            role="user",
        )
        req = self._make_llm_request()
        result = callback(ctx, req)
        assert result is None
        req.append_instructions.assert_not_called()

    def test_callback_no_results_empty_store(self, empty_store):
        callback = create_example_callback(empty_store, k=2)
        ctx = self._make_context("hello")
        req = self._make_llm_request()
        result = callback(ctx, req)
        assert result is None
        req.append_instructions.assert_not_called()

    def test_callback_custom_prefix(self, store):
        callback = create_example_callback(
            store, k=1, instruction_prefix="\n\nCustom prefix:\n"
        )
        ctx = self._make_context("math")
        req = self._make_llm_request()
        callback(ctx, req)
        instruction = req.append_instructions.call_args[0][0][0]
        assert "Custom prefix:" in instruction

    def test_callback_with_category(self, store):
        callback = create_example_callback(store, k=10, category="greeting")
        ctx = self._make_context("anything")
        req = self._make_llm_request()
        callback(ctx, req)
        instruction = req.append_instructions.call_args[0][0][0]
        # Should only contain greeting examples
        assert "Hello" in instruction or "Goodbye" in instruction
