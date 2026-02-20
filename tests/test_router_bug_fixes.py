"""
Test suite to verify the critical bug fixes in ModelRouter.

This test suite specifically validates the fixes for:
1. The "Polite User" Penalty - greetings with actual questions
2. The "Substring Match" Bug - "csv" triggering "vs" match
3. Inconsistent Reasoning Output - logging "none" when matches exist
4. Overly Broad Meta-Comments - "help" in longer queries
"""

import sys
sys.path.insert(0, 'backend')

import pytest
from services.model_router import ModelRouter


class TestRouterBugFixes:
    """Test suite for verifying critical bug fixes in ModelRouter."""
    
    @pytest.fixture
    def router(self):
        """Create a ModelRouter instance for testing."""
        return ModelRouter()
    
    # Bug Fix 1: The "Polite User" Penalty
    def test_greeting_with_actual_question_not_ood(self, router):
        """Verify that 'Hi, how do I reset my password?' is NOT treated as OOD."""
        result = router.classify_query("Hi, how do I reset my password?")
        
        # Should NOT skip retrieval - this is a real question
        assert result.skip_retrieval is False, "Should not skip retrieval for greeting + question"
        # Should route to complex model due to "how" keyword
        assert result.category == "complex"
        assert result.model_name == "llama-3.3-70b-versatile"
    
    def test_hello_with_technical_question(self, router):
        """Verify that 'Hello, what is the pricing for enterprise plan?' works correctly."""
        result = router.classify_query("Hello, what is the pricing for enterprise plan?")
        
        # Should NOT skip retrieval
        assert result.skip_retrieval is False
        # Should be simple (no complex keywords)
        assert result.category == "simple"
    
    def test_thanks_with_followup(self, router):
        """Verify that 'Thanks! Can you explain the API rate limits?' is not OOD."""
        result = router.classify_query("Thanks! Can you explain the API rate limits?")
        
        # Should NOT skip retrieval
        assert result.skip_retrieval is False
        # Should route to complex due to "explain"
        assert result.category == "complex"
    
    def test_standalone_greeting_is_ood(self, router):
        """Verify that standalone 'Hi' or 'Hello' IS treated as OOD."""
        result = router.classify_query("Hi")
        assert result.skip_retrieval is True
        
        result = router.classify_query("Hello!")
        assert result.skip_retrieval is True
        
        result = router.classify_query("Thanks")
        assert result.skip_retrieval is True
    
    # Bug Fix 2: The "Substring Match" Bug
    def test_csv_does_not_trigger_vs_match(self, router):
        """Verify that queries with 'csv' don't trigger comparison word 'vs'."""
        result = router.classify_query("How do I export data to CSV format?")
        
        # Should NOT match comparison words
        assert "comparison_words" not in result.rule_triggered
        # Should route to complex due to "how"
        assert result.category == "complex"
        assert result.rule_triggered == "complex_keyword"
    
    def test_devs_does_not_trigger_vs_match(self, router):
        """Verify that 'devs' doesn't trigger 'vs' match."""
        result = router.classify_query("What tools do devs use?")
        
        # Should be simple (no complex keywords, no comparison)
        assert result.category == "simple"
        assert "comparison" not in result.reasoning.lower()
    
    def test_actual_vs_triggers_comparison(self, router):
        """Verify that actual 'vs' or 'versus' DOES trigger comparison."""
        result = router.classify_query("Python vs JavaScript")
        
        # Should route to complex due to comparison
        assert result.category == "complex"
        assert result.rule_triggered == "comparison_words"
        assert "vs" in result.reasoning
    
    def test_versus_triggers_comparison(self, router):
        """Verify that 'versus' triggers comparison."""
        result = router.classify_query("Enterprise versus Pro plan")
        
        assert result.category == "complex"
        assert result.rule_triggered == "comparison_words"
        assert "versus" in result.reasoning
    
    # Bug Fix 3: Inconsistent Reasoning Output
    def test_reasoning_never_says_none_when_match_exists(self, router):
        """Verify that reasoning doesn't say 'none' when keywords are matched."""
        result = router.classify_query("Why is the system slow?")
        
        # Should match "why" keyword
        assert result.category == "complex"
        assert "why" in result.reasoning.lower()
        assert "none" not in result.reasoning.lower()
    
    def test_comparison_reasoning_shows_actual_word(self, router):
        """Verify that comparison word reasoning shows the actual matched word."""
        result = router.classify_query("Which is better for my use case?")
        
        assert result.category == "complex"
        assert result.rule_triggered == "comparison_words"
        assert "better" in result.reasoning
        assert "none" not in result.reasoning.lower()
    
    # Bug Fix 4: Overly Broad Meta-Comments
    def test_help_in_longer_query_not_meta(self, router):
        """Verify that 'help' in a longer technical query is NOT treated as meta-comment."""
        result = router.classify_query("I need help configuring my firewall settings")
        
        # Should NOT skip retrieval
        assert result.skip_retrieval is False
        # Should be simple (no complex keywords)
        assert result.category == "simple"
        assert result.rule_triggered != "ood_filter"
    
    def test_help_with_technical_context(self, router):
        """Verify that 'Can you help me install Docker?' is not OOD."""
        result = router.classify_query("Can you help me install Docker?")
        
        # Should NOT skip retrieval
        assert result.skip_retrieval is False
        assert result.rule_triggered != "ood_filter"
    
    def test_standalone_help_is_meta(self, router):
        """Verify that standalone 'help' IS treated as meta-comment."""
        result = router.classify_query("help")
        
        # Should skip retrieval
        assert result.skip_retrieval is True
        assert result.rule_triggered == "ood_filter"
    
    def test_short_help_query_is_meta(self, router):
        """Verify that very short 'help' queries are treated as meta."""
        result = router.classify_query("help me")
        
        # Should skip retrieval (3 words or less)
        assert result.skip_retrieval is True
    
    # Additional edge cases
    def test_word_boundary_enforcement(self, router):
        """Verify that word boundaries are properly enforced."""
        # "showing" should not match "how"
        result = router.classify_query("What is showing in the dashboard?")
        assert result.category == "simple"
        
        # "whoever" should not match "who"
        result = router.classify_query("Whoever configured this did great")
        assert result.category == "simple"
    
    def test_multi_word_comparison_phrases(self, router):
        """Verify that multi-word comparison phrases like 'compared to' work."""
        result = router.classify_query("How does this compared to the old version?")
        
        # Should match "how" first (complex keyword takes precedence)
        assert result.category == "complex"
        # But "compared to" should also be detectable
        assert result.rule_triggered in ["complex_keyword", "comparison_words"]
