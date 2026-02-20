"""
Unit tests for ModelRouter class.

Tests the decision tree logic for query classification and model routing.
"""

import sys
sys.path.insert(0, 'backend')

import pytest
from services.model_router import ModelRouter, Classification


class TestModelRouter:
    """Test suite for ModelRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create a ModelRouter instance for testing."""
        return ModelRouter()
    
    # Rule 0: OOD Filter Tests
    
    def test_greeting_hello(self, router):
        """Test that 'hello' is classified as simple with skip_retrieval."""
        result = router.classify_query("hello")
        assert result.category == ModelRouter.SIMPLE
        assert result.model_name == ModelRouter.SIMPLE_MODEL
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_greeting_hi(self, router):
        """Test that 'hi' is classified as simple with skip_retrieval."""
        result = router.classify_query("hi")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_greeting_hey(self, router):
        """Test that 'hey' is classified as simple with skip_retrieval."""
        result = router.classify_query("hey")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_greeting_thanks(self, router):
        """Test that 'thanks' is classified as simple with skip_retrieval."""
        result = router.classify_query("thanks")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_greeting_thank_you(self, router):
        """Test that 'thank you' is classified as simple with skip_retrieval."""
        result = router.classify_query("thank you")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_greeting_with_punctuation(self, router):
        """Test that 'Hello!' is classified as simple with skip_retrieval."""
        result = router.classify_query("Hello!")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_meta_comment_who_are_you(self, router):
        """Test that 'who are you' is classified as simple with skip_retrieval."""
        result = router.classify_query("who are you")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_meta_comment_what_can_you_do(self, router):
        """Test that 'what can you do' is classified as simple with skip_retrieval."""
        result = router.classify_query("what can you do")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_meta_comment_help(self, router):
        """Test that 'help' is classified as simple with skip_retrieval."""
        result = router.classify_query("help")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    # Rule 1: Complex Keywords Tests
    
    def test_complex_keyword_why(self, router):
        """Test that query with 'why' is classified as complex."""
        result = router.classify_query("Why is the sky blue?")
        assert result.category == ModelRouter.COMPLEX
        assert result.model_name == ModelRouter.COMPLEX_MODEL
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
        assert "why" in result.reasoning.lower()
    
    def test_complex_keyword_how(self, router):
        """Test that query with 'how' is classified as complex."""
        result = router.classify_query("How do I configure workflows?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    def test_complex_keyword_explain(self, router):
        """Test that query with 'explain' is classified as complex."""
        result = router.classify_query("Explain the pricing model")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    def test_complex_keyword_compare(self, router):
        """Test that query with 'compare' is classified as complex."""
        result = router.classify_query("Compare the Pro and Enterprise plans")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    def test_complex_keyword_analyze(self, router):
        """Test that query with 'analyze' is classified as complex."""
        result = router.classify_query("Analyze the differences between plans")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    def test_complex_keyword_difference(self, router):
        """Test that query with 'difference' is classified as complex."""
        result = router.classify_query("What is the difference between Pro and Enterprise?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    def test_complex_keyword_relationship(self, router):
        """Test that query with 'relationship' is classified as complex."""
        result = router.classify_query("What is the relationship between tasks and projects?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "complex_keyword"
    
    # Rule 2: Query Length Tests
    
    def test_query_length_exactly_15_words(self, router):
        """Test that query with exactly 15 words is classified as simple."""
        query = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        result = router.classify_query(query)
        # Should not trigger length rule (needs > 15)
        assert result.category == ModelRouter.SIMPLE
        assert result.rule_triggered == "default"
    
    def test_query_length_16_words(self, router):
        """Test that query with 16 words is classified as complex."""
        query = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
        result = router.classify_query(query)
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "query_length"
        assert "16 words" in result.reasoning
    
    def test_query_length_long_sentence(self, router):
        """Test that a long query without complex keywords is classified as complex."""
        query = "What is the price of the Pro plan and does it include all the features listed on the website?"
        word_count = len(query.split())
        result = router.classify_query(query)
        if word_count > 15:
            assert result.category == ModelRouter.COMPLEX
            assert result.rule_triggered == "query_length"
    
    # Rule 3: Multiple Questions Tests
    
    def test_single_question_mark(self, router):
        """Test that query with single question mark doesn't trigger multiple questions rule."""
        result = router.classify_query("What is the price?")
        # Should not trigger multiple questions rule
        assert result.rule_triggered != "multiple_questions"
    
    def test_two_question_marks(self, router):
        """Test that query with two question marks is classified as complex."""
        result = router.classify_query("What is the price? Is it monthly?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "multiple_questions"
        assert "2" in result.reasoning
    
    def test_three_question_marks(self, router):
        """Test that query with three question marks is classified as complex."""
        result = router.classify_query("What is pricing? What about features? Any discounts?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "multiple_questions"
    
    # Rule 4: Comparison Words Tests
    
    def test_comparison_word_versus(self, router):
        """Test that query with 'versus' is classified as complex."""
        result = router.classify_query("Pro versus Enterprise")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "comparison_words"
    
    def test_comparison_word_vs(self, router):
        """Test that query with 'vs' is classified as complex."""
        result = router.classify_query("Pro vs Enterprise")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "comparison_words"
    
    def test_comparison_word_better(self, router):
        """Test that query with 'better' is classified as complex."""
        result = router.classify_query("Which plan is better?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "comparison_words"
    
    def test_comparison_word_worse(self, router):
        """Test that query with 'worse' is classified as complex."""
        result = router.classify_query("Is the Basic plan worse?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        assert result.rule_triggered == "comparison_words"
    
    def test_comparison_phrase_compared_to(self, router):
        """Test that query with 'compared to' is classified as complex."""
        result = router.classify_query("How does Pro compare compared to Enterprise?")
        assert result.category == ModelRouter.COMPLEX
        assert result.skip_retrieval is False
        # Note: This triggers complex_keyword ("how", "compare") before comparison_words
        # which is correct per the decision tree priority
        assert result.rule_triggered in ["complex_keyword", "comparison_words"]
    
    # Rule 5: Default (Simple) Tests
    
    def test_simple_factual_question(self, router):
        """Test that simple factual question is classified as simple."""
        result = router.classify_query("What is the Pro plan price?")
        assert result.category == ModelRouter.SIMPLE
        assert result.model_name == ModelRouter.SIMPLE_MODEL
        assert result.skip_retrieval is False
        assert result.rule_triggered == "default"
    
    def test_simple_list_request(self, router):
        """Test that simple list request is classified as simple."""
        result = router.classify_query("List keyboard shortcuts")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is False
        assert result.rule_triggered == "default"
    
    def test_simple_short_query(self, router):
        """Test that short query without triggers is classified as simple."""
        result = router.classify_query("Pricing?")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is False
        assert result.rule_triggered == "default"
    
    # Edge Cases
    
    def test_empty_query(self, router):
        """Test that empty query is handled gracefully."""
        result = router.classify_query("")
        assert result.category == ModelRouter.SIMPLE
        assert result.model_name == ModelRouter.SIMPLE_MODEL
    
    def test_whitespace_only_query(self, router):
        """Test that whitespace-only query is handled gracefully."""
        result = router.classify_query("   ")
        assert result.category == ModelRouter.SIMPLE
        assert result.model_name == ModelRouter.SIMPLE_MODEL
    
    def test_case_insensitive_keywords(self, router):
        """Test that keyword detection is case-insensitive."""
        result = router.classify_query("WHY is this happening?")
        assert result.category == ModelRouter.COMPLEX
        assert result.rule_triggered == "complex_keyword"
    
    def test_case_insensitive_greetings(self, router):
        """Test that greeting detection is case-insensitive."""
        result = router.classify_query("HELLO")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    # Decision Tree Priority Tests
    
    def test_ood_filter_overrides_complex_keywords(self, router):
        """Test that OOD filter takes priority over complex keywords."""
        # "help" is both a greeting and could be interpreted as needing explanation
        result = router.classify_query("help")
        assert result.rule_triggered == "ood_filter"
        assert result.skip_retrieval is True
    
    def test_complex_keyword_overrides_length(self, router):
        """Test that complex keywords take priority over query length."""
        # Short query with complex keyword
        result = router.classify_query("Why?")
        assert result.rule_triggered == "complex_keyword"
        assert result.category == ModelRouter.COMPLEX
    
    def test_complex_keyword_overrides_comparison(self, router):
        """Test that complex keywords take priority over comparison words."""
        # Query with both complex keyword and comparison word
        result = router.classify_query("Explain Pro vs Enterprise")
        assert result.rule_triggered == "complex_keyword"
        assert result.category == ModelRouter.COMPLEX
    
    # Boundary Tests
    
    def test_partial_keyword_match_not_triggered(self, router):
        """Test that partial keyword matches don't trigger complex classification."""
        # "somewhere" contains "where" but shouldn't match "why"
        result = router.classify_query("Is it somewhere?")
        # Should not trigger complex keyword rule
        assert result.rule_triggered != "complex_keyword"
    
    def test_greeting_as_part_of_sentence(self, router):
        """Test that greeting in middle of sentence doesn't trigger OOD filter."""
        result = router.classify_query("I want to say hello to the team")
        # Should not trigger OOD filter since "hello" is not at the start
        assert result.rule_triggered != "ood_filter"
    
    # Real-World Examples from Design Doc
    
    def test_design_example_greeting(self, router):
        """Test design doc example: 'Hello!' → Simple + skip retrieval."""
        result = router.classify_query("Hello!")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is True
    
    def test_design_example_simple_pricing(self, router):
        """Test design doc example: 'What is the Pro plan price?' → Simple."""
        result = router.classify_query("What is the Pro plan price?")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is False
    
    def test_design_example_complex_how(self, router):
        """Test design doc example: 'How do I configure custom workflows?' → Complex."""
        result = router.classify_query("How do I configure custom workflows?")
        assert result.category == ModelRouter.COMPLEX
    
    def test_design_example_complex_compare(self, router):
        """Test design doc example: 'Compare Enterprise vs Pro features' → Complex."""
        result = router.classify_query("Compare Enterprise vs Pro features")
        assert result.category == ModelRouter.COMPLEX
    
    def test_design_example_simple_list(self, router):
        """Test design doc example: 'List keyboard shortcuts' → Simple."""
        result = router.classify_query("List keyboard shortcuts")
        assert result.category == ModelRouter.SIMPLE
        assert result.skip_retrieval is False
    
    # Classification Dataclass Tests
    
    def test_classification_has_all_fields(self, router):
        """Test that Classification dataclass has all required fields."""
        result = router.classify_query("Test query")
        assert hasattr(result, 'category')
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'skip_retrieval')
        assert hasattr(result, 'rule_triggered')
    
    def test_classification_reasoning_not_empty(self, router):
        """Test that classification always includes reasoning."""
        result = router.classify_query("What is pricing?")
        assert result.reasoning
        assert len(result.reasoning) > 0
    
    def test_classification_model_name_valid(self, router):
        """Test that model_name is always one of the two valid models."""
        queries = [
            "Hello",
            "What is pricing?",
            "How do I configure workflows?",
            "Pro vs Enterprise"
        ]
        for query in queries:
            result = router.classify_query(query)
            assert result.model_name in [ModelRouter.SIMPLE_MODEL, ModelRouter.COMPLEX_MODEL]
