"""Functional integration tests for Langchain adapter.

Tests that LanceDBExampleSelector integrates with Langchain's
prompt template system.
"""

import pytest

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)

from example_selector.langchain.selector import LanceDBExampleSelector


class TestFewShotPromptTemplateIntegration:
    """Test integration with Langchain's FewShotPromptTemplate (string prompts)."""

    def test_with_few_shot_prompt_template(self, store):
        selector = LanceDBExampleSelector(store=store, k=2)
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        )
        few_shot = FewShotPromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            prefix="Here are some examples:",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"],
        )
        result = few_shot.format(input="say hello")
        assert "Here are some examples:" in result
        assert "Input:" in result
        assert "Output:" in result

    def test_selected_examples_are_relevant(self, store):
        selector = LanceDBExampleSelector(store=store, k=1)
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        )
        few_shot = FewShotPromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            prefix="Examples:",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"],
        )
        result = few_shot.format(input="capital of Germany")
        # Should select the geography example about France
        assert "capital" in result.lower() or "france" in result.lower() or "paris" in result.lower()


class TestFewShotChatMessagePromptTemplateIntegration:
    """Test integration with Langchain's chat message prompt templates."""

    def test_with_chat_prompt_template(self, store):
        selector = LanceDBExampleSelector(store=store, k=2)
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        messages = few_shot.format_messages(input="say hello")
        assert len(messages) >= 2  # At least 1 example = 2 messages
        # Alternating human/ai messages
        roles = [m.type for m in messages]
        for i in range(0, len(roles), 2):
            assert roles[i] == "human"
        for i in range(1, len(roles), 2):
            assert roles[i] == "ai"

    def test_composed_in_chat_prompt(self, store):
        selector = LanceDBExampleSelector(store=store, k=2)
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                few_shot,
                ("human", "{input}"),
            ]
        )
        messages = final_prompt.format_messages(input="greet me")
        # Should have: system + example pairs + human
        assert messages[0].type == "system"
        assert messages[-1].type == "human"
        assert messages[-1].content == "greet me"
        # At least 1 example pair between system and final human
        assert len(messages) >= 4  # system + 2 example msgs + human

    def test_no_examples_in_empty_store(self, empty_store):
        selector = LanceDBExampleSelector(store=empty_store, k=2)
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        messages = few_shot.format_messages(input="hello")
        assert messages == []

    def test_category_filtered_examples_in_chat(self, store):
        selector = LanceDBExampleSelector(
            store=store, k=10, category="greeting"
        )
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        messages = few_shot.format_messages(input="anything")
        # Only greeting examples (2 in sample data) -> 4 messages max
        assert len(messages) <= 4


class TestEndToEnd:
    """Full pipeline tests."""

    def test_add_then_select(self, empty_store):
        selector = LanceDBExampleSelector(store=empty_store, k=2)
        selector.add_example({"input": "What is 2+2?", "output": "4"})
        selector.add_example({"input": "What is 3+3?", "output": "6"})
        selector.add_example({"input": "Tell a joke", "output": "Why did..."})

        results = selector.select_examples({"input": "math addition"})
        assert len(results) >= 1
        # Math examples should be more relevant
        inputs = [r["input"] for r in results]
        assert any("2+2" in i or "3+3" in i for i in inputs)
