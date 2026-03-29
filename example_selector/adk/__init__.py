"""Google ADK adapter for dynamic example selection."""

from example_selector.adk.agent import ExampleSelectorAgent
from example_selector.adk.plugin import ExampleSelectorPlugin
from example_selector.adk.helpers import create_example_callback

__all__ = ["ExampleSelectorAgent", "ExampleSelectorPlugin", "create_example_callback"]
