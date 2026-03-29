"""ExampleSelectorPlugin: runner-level example injection for all agents."""

from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin

from example_selector.adk.agent import _default_format
from example_selector.core.store import ExampleStore


class ExampleSelectorPlugin(BasePlugin):
    """Plugin that injects relevant examples into every agent's LLM request.

    Attach to a ``Runner`` or ``App`` to apply example injection globally to
    all agents without modifying any individual agent definition.

    Use this when all agents in the runner/app share the same example store.
    For per-agent control, prefer :class:`~example_selector.adk.ExampleSelectorAgent`.

    Args:
        store: The :class:`ExampleStore` to search for examples.
        k: Number of top examples to retrieve per request (default 5).
        category: Optional category filter applied to all agents.
    """

    def __init__(
        self,
        store: ExampleStore,
        k: int = 5,
        category: Optional[str] = None,
    ) -> None:
        super().__init__(name="example_selector")
        self._store = store
        self._k = k
        self._category = category

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        """Append relevant examples to the system instruction before model call."""
        if callback_context.user_content is None:
            return None
        parts = callback_context.user_content.parts
        if not parts or not parts[0].text:
            return None

        query = parts[0].text
        examples = self._store.search(query, k=self._k, category=self._category)
        if not examples:
            return None

        llm_request.append_instructions([_default_format(examples)])
        return None
