"""ExampleSelectorAgent: LlmAgent with automatic example injection."""

from __future__ import annotations

from typing import Callable, Optional, Union

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from example_selector.core.store import ExampleStore


def _default_format(examples: list[dict]) -> str:
    lines = ["Here are relevant examples to guide your response style:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  User: {ex['input_text']}")
        lines.append(f"  Assistant: {ex['output_text']}")
        lines.append("")
    return "\n".join(lines)


class ExampleSelectorAgent(LlmAgent):
    """LlmAgent that automatically injects relevant examples into its instruction.

    Drop-in replacement for ``LlmAgent``. Accepts all native LlmAgent parameters
    plus the example-selector parameters below. At runtime, ``canonical_instruction``
    is overridden to append semantically similar examples to the agent's system
    instruction before the request is sent to the LLM.

    All native LlmAgent features (tools, callbacks, sub-agents, plugins, state
    injection) continue to work unchanged.

    Args:
        example_store: The :class:`ExampleStore` to search for examples.
        example_k: Number of top examples to retrieve (default 5).
        example_category: Optional category filter for the search.
        example_format: Callable that formats a list of example dicts into a
            string, or the string ``"default"`` to use the built-in formatter.
    """

    example_store: ExampleStore
    example_k: int = 5
    example_category: Optional[str] = None
    example_format: Union[str, Callable[[list[dict]], str]] = "default"

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    async def canonical_instruction(self, ctx: ReadonlyContext) -> tuple[str, bool]:
        """Return the agent instruction enriched with relevant examples."""
        base_instruction, bypass = await super().canonical_instruction(ctx)

        query = self._extract_query(ctx)
        if not query:
            return base_instruction, bypass

        examples = self.example_store.search(
            query, k=self.example_k, category=self.example_category
        )
        if not examples:
            return base_instruction, bypass

        formatted = self._format_examples(examples)
        return f"{base_instruction}\n\n{formatted}", bypass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query(self, ctx: ReadonlyContext) -> Optional[str]:
        """Extract the user query text from the context."""
        if ctx.user_content is None:
            return None
        parts = ctx.user_content.parts
        if not parts:
            return None
        text = parts[0].text
        return text if text else None

    def _format_examples(self, examples: list[dict]) -> str:
        if callable(self.example_format):
            return self.example_format(examples)
        return _default_format(examples)
