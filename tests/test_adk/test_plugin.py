"""Tests for ExampleSelectorPlugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from google.adk.plugins import BasePlugin
from google.genai import types

from example_selector.adk.plugin import ExampleSelectorPlugin


def _make_callback_context(text: str | None = "hello"):
    ctx = MagicMock()
    if text is None:
        ctx.user_content = None
    else:
        ctx.user_content = types.Content(
            parts=[types.Part(text=text)], role="user"
        )
    return ctx


def _make_llm_request():
    req = MagicMock()
    req.append_instructions = MagicMock()
    return req


class TestExampleSelectorPlugin:
    def test_is_base_plugin_subclass(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        assert isinstance(plugin, BasePlugin)

    def test_plugin_name(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        assert plugin.name == "example_selector"

    def test_default_params(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        assert plugin._k == 5
        assert plugin._category is None

    @pytest.mark.asyncio
    async def test_appends_instructions_when_examples_found(self, store):
        plugin = ExampleSelectorPlugin(store=store, k=2)
        ctx = _make_callback_context("greet someone")
        req = _make_llm_request()
        result = await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert result is None
        req.append_instructions.assert_called_once()
        instruction = req.append_instructions.call_args[0][0][0]
        assert "Example" in instruction

    @pytest.mark.asyncio
    async def test_skips_when_no_user_content(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        ctx = _make_callback_context(None)
        req = _make_llm_request()
        result = await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert result is None
        req.append_instructions.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_empty_parts(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        ctx = MagicMock()
        ctx.user_content = types.Content(parts=[], role="user")
        req = _make_llm_request()
        result = await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert result is None
        req.append_instructions.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_text_part(self, store):
        plugin = ExampleSelectorPlugin(store=store)
        ctx = MagicMock()
        ctx.user_content = types.Content(
            parts=[types.Part(inline_data=types.Blob(data=b"img", mime_type="image/png"))],
            role="user",
        )
        req = _make_llm_request()
        result = await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert result is None
        req.append_instructions.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_empty_store(self, empty_store):
        plugin = ExampleSelectorPlugin(store=empty_store, k=5)
        ctx = _make_callback_context("hello")
        req = _make_llm_request()
        result = await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        assert result is None
        req.append_instructions.assert_not_called()

    @pytest.mark.asyncio
    async def test_category_filter(self, store):
        plugin = ExampleSelectorPlugin(store=store, k=10, category="greeting")
        ctx = _make_callback_context("anything")
        req = _make_llm_request()
        await plugin.before_model_callback(callback_context=ctx, llm_request=req)
        instruction = req.append_instructions.call_args[0][0][0]
        assert "Hello" in instruction or "Goodbye" in instruction
