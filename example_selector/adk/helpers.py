"""Helper functions for ADK integration."""

from __future__ import annotations

from typing import Optional

from example_selector.core.store import ExampleStore


def create_example_callback(
    store: ExampleStore,
    k: int = 5,
    category: Optional[str] = None,
    instruction_prefix: str = "\n\nRelevant examples:\n",
):
    """Return a ``before_model_callback`` that injects examples into the LLM request.

    Secondary/escape-hatch alternative to :class:`~example_selector.adk.ExampleSelectorAgent`
    for users who need to keep vanilla ``LlmAgent`` (e.g. for compatibility reasons).

    Args:
        store: The :class:`~example_selector.core.store.ExampleStore` to search.
        k: Number of top examples to retrieve (default 5).
        category: Optional category filter.
        instruction_prefix: Text prepended before the formatted examples block.

    Returns:
        A synchronous ``before_model_callback`` that appends example instructions
        to the ``LlmRequest`` and returns ``None`` to continue the LLM call.
    """

    def _callback(callback_context, llm_request):
        parts = callback_context.user_content.parts
        if not parts or not parts[0].text:
            return None

        query = parts[0].text
        results = store.search(query, k=k, category=category)

        if not results:
            return None

        lines = [instruction_prefix]
        for i, r in enumerate(results, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {r['input_text']}")
            lines.append(f"  Output: {r['output_text']}")
            lines.append("")

        llm_request.append_instructions(["\n".join(lines)])
        return None  # Continue with LLM call

    return _callback
