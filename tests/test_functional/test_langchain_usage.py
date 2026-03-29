"""
Functional test: Langchain usage example.

Demonstrates how to use example-selector with Langchain chains and agents.
This mirrors real-world usage patterns and tests the full integration.
"""

import tempfile

import pytest

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from example_selector import ExampleStore
from example_selector.langchain import LanceDBExampleSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qa_store():
    """A store simulating Q&A few-shot examples."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ExampleStore(db_path=tmp, table_name="qa")
        store.add_examples([
            {
                "input_text": "What is photosynthesis?",
                "output_text": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs primarily in the leaves using chlorophyll.",
                "category": "science",
            },
            {
                "input_text": "Explain gravity in simple terms",
                "output_text": "Gravity is the force that pulls objects toward each other. On Earth, it's what keeps us on the ground and makes things fall when we drop them.",
                "category": "science",
            },
            {
                "input_text": "What caused World War I?",
                "output_text": "WWI was caused by a combination of factors: alliance systems, militarism, imperialism, and nationalism. The assassination of Archduke Franz Ferdinand was the immediate trigger.",
                "category": "history",
            },
            {
                "input_text": "Who was Cleopatra?",
                "output_text": "Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt. She was known for her intelligence, political acumen, and relationships with Julius Caesar and Mark Antony.",
                "category": "history",
            },
            {
                "input_text": "How do computers store data?",
                "output_text": "Computers store data in binary format (0s and 1s) using various storage devices like SSDs, HDDs, and RAM. Data is organized in bytes, with each byte containing 8 bits.",
                "category": "technology",
            },
        ])
        yield store


# ---------------------------------------------------------------------------
# Usage Example 1: FewShotChatMessagePromptTemplate (primary pattern)
# ---------------------------------------------------------------------------

class TestLangchainChatPromptUsage:
    """
    PRIMARY USAGE PATTERN: FewShotChatMessagePromptTemplate

    This is the recommended way for chat-based Langchain chains.
    """

    def test_basic_chat_prompt_with_examples(self, qa_store):
        """Example: Q&A chain with dynamically selected examples."""
        selector = LanceDBExampleSelector(store=qa_store, k=2)

        # Define how each example should be formatted as messages
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        # Create the few-shot template
        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )

        # Compose into a full chat prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable tutor. Answer questions clearly and concisely."),
            few_shot,
            ("human", "{input}"),
        ])

        # Format for a science question
        messages = final_prompt.format_messages(input="How does the water cycle work?")

        assert messages[0].type == "system"
        assert messages[0].content == "You are a knowledgeable tutor. Answer questions clearly and concisely."
        assert messages[-1].type == "human"
        assert messages[-1].content == "How does the water cycle work?"
        # Should have at least 1 example pair between system and user message
        assert len(messages) >= 4

    def test_category_filtered_chat_prompt(self, qa_store):
        """Example: Only showing history examples."""
        selector = LanceDBExampleSelector(
            store=qa_store, k=5, category="history"
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a history expert."),
            few_shot,
            ("human", "{input}"),
        ])

        messages = prompt.format_messages(input="Tell me about ancient Rome")
        # System + up to 2 history examples (4 msgs) + user = max 7
        example_messages = messages[1:-1]  # Between system and user
        # Only history examples (2 exist)
        assert len(example_messages) <= 4

    def test_custom_keys(self, qa_store):
        """Example: Using custom key names for input/output."""
        selector = LanceDBExampleSelector(
            store=qa_store, k=1, input_key="question", output_key="answer"
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}"),
        ])

        few_shot = FewShotChatMessagePromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            input_variables=["question"],
        )

        messages = few_shot.format_messages(question="What is DNA?")
        assert len(messages) >= 2
        # Should be a human/ai pair
        assert messages[0].type == "human"
        assert messages[1].type == "ai"


# ---------------------------------------------------------------------------
# Usage Example 2: FewShotPromptTemplate (string prompts)
# ---------------------------------------------------------------------------

class TestLangchainStringPromptUsage:
    """
    STRING PROMPT PATTERN: FewShotPromptTemplate

    For non-chat chains or when you need a single formatted string.
    """

    def test_string_prompt_with_examples(self, qa_store):
        """Example: String-based prompt with dynamic examples."""
        selector = LanceDBExampleSelector(store=qa_store, k=2)

        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Question: {input}\nAnswer: {output}",
        )

        few_shot = FewShotPromptTemplate(
            example_selector=selector,
            example_prompt=example_prompt,
            prefix="Answer the following question based on these examples:\n",
            suffix="\nQuestion: {input}\nAnswer:",
            input_variables=["input"],
        )

        result = few_shot.format(input="What is evolution?")

        assert "Answer the following question based on these examples:" in result
        assert "Question: What is evolution?" in result
        assert "Answer:" in result
        # Should contain at least one example
        assert result.count("Question:") >= 2  # At least 1 example + the actual question


# ---------------------------------------------------------------------------
# Usage Example 3: from_examples factory
# ---------------------------------------------------------------------------

class TestLangchainFromExamples:
    """
    BOOTSTRAP PATTERN: Create selector pre-populated with examples.
    """

    def test_from_examples_factory(self):
        """Example: Quick setup with from_examples."""
        with tempfile.TemporaryDirectory() as tmp:
            selector = LanceDBExampleSelector.from_examples(
                examples=[
                    {"input": "Summarize this article", "output": "The article discusses..."},
                    {"input": "Translate to French", "output": "Bonjour, comment..."},
                    {"input": "Fix the grammar", "output": "Here is the corrected text..."},
                ],
                db_path=tmp,
                table_name="tasks",
                k=2,
            )

            results = selector.select_examples({"input": "Please translate this"})
            assert len(results) >= 1
            # Translation example should be highly ranked
            inputs = [r["input"] for r in results]
            assert any("translate" in i.lower() for i in inputs)


# ---------------------------------------------------------------------------
# Usage Example 4: Dynamic updates via add_example
# ---------------------------------------------------------------------------

class TestLangchainDynamicUpdates:
    """
    DYNAMIC PATTERN: Adding examples through the Langchain interface.
    """

    def test_add_example_via_selector(self, qa_store):
        """Example: Adding a new example through the selector."""
        selector = LanceDBExampleSelector(store=qa_store, k=5)

        initial_count = qa_store.count()

        selector.add_example({
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed.",
            "category": "technology",
        })

        assert qa_store.count() == initial_count + 1

        # The new example should be findable
        results = selector.select_examples({"input": "Tell me about ML and AI"})
        inputs = [r["input"] for r in results]
        assert any("machine learning" in i.lower() for i in inputs)


# ---------------------------------------------------------------------------
# Usage Example 5: example_keys filter
# ---------------------------------------------------------------------------

class TestLangchainExampleKeysFilter:
    """Test that example_keys properly filters returned dictionaries."""

    def test_only_input_output_keys(self, qa_store):
        """Only include specified keys in returned examples."""
        selector = LanceDBExampleSelector(
            store=qa_store, k=2, example_keys=["input", "output"]
        )
        results = selector.select_examples({"input": "science question"})
        for r in results:
            assert set(r.keys()) == {"input", "output"}
            assert "category" not in r
            assert "metadata" not in r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestLangchainEdgeCases:
    def test_empty_store(self):
        """Empty store should return no examples."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ExampleStore(db_path=tmp, table_name="empty")
            selector = LanceDBExampleSelector(store=store, k=5)
            results = selector.select_examples({"input": "hello"})
            assert results == []

    def test_add_example_with_empty_input(self, qa_store):
        """Adding an example with empty input should work but may not be useful."""
        selector = LanceDBExampleSelector(store=qa_store, k=5)
        eid = selector.add_example({"input": "", "output": "empty input"})
        assert isinstance(eid, str)

    def test_select_with_missing_input_key(self, qa_store):
        """When input_key is not in input_variables, should use fallback."""
        selector = LanceDBExampleSelector(store=qa_store, k=2, input_key="query")
        # "query" key not present, falls back to joining all values
        results = selector.select_examples({"question": "science", "topic": "biology"})
        assert len(results) >= 1

    def test_very_short_query(self, qa_store):
        """Very short queries should still work."""
        selector = LanceDBExampleSelector(store=qa_store, k=2)
        results = selector.select_examples({"input": "a"})
        assert isinstance(results, list)

    def test_html_in_examples(self):
        """HTML content should be stored and retrieved correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            store = ExampleStore(db_path=tmp, table_name="html")
            store.add_example(
                input_text="Format this as HTML",
                output_text="<div class='container'><h1>Title</h1><p>Content</p></div>",
            )
            selector = LanceDBExampleSelector(store=store, k=1)
            results = selector.select_examples({"input": "create HTML layout"})
            assert len(results) == 1
            assert "<div" in results[0]["output"]

    def test_multiple_selectors_same_store(self, qa_store):
        """Multiple selectors can share one store with different configs."""
        s1 = LanceDBExampleSelector(store=qa_store, k=1, category="science")
        s2 = LanceDBExampleSelector(store=qa_store, k=1, category="history")

        r1 = s1.select_examples({"input": "question"})
        r2 = s2.select_examples({"input": "question"})

        assert len(r1) >= 1
        assert len(r2) >= 1
        # They should return different examples
        assert r1[0]["input"] != r2[0]["input"]
