"""Unit tests for OutputEvaluator."""
import sys
sys.path.insert(0, 'backend')

import pytest
from services.output_evaluator import OutputEvaluator
from models.chunk import Chunk, ScoredChunk


@pytest.fixture
def evaluator():
    """Create OutputEvaluator instance."""
    return OutputEvaluator()


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunk1 = Chunk(
        chunk_id="doc1_1_0",
        text="ClearPath offers three pricing plans: Basic at $10/month, Pro at $25/month, and Enterprise with custom pricing.",
        document_name="pricing_guide.pdf",
        page_number=1,
        token_count=20
    )
    
    chunk2 = Chunk(
        chunk_id="doc1_2_0",
        text="The Pro plan includes advanced features like custom workflows, API access, and priority support.",
        document_name="pricing_guide.pdf",
        page_number=2,
        token_count=18
    )
    
    return [
        ScoredChunk(chunk=chunk1, relevance_score=0.85),
        ScoredChunk(chunk=chunk2, relevance_score=0.72)
    ]


class TestNoContextDetection:
    """Tests for no-context detection."""
    
    def test_no_context_with_answer(self, evaluator):
        """Should flag when chunks=0 and LLM provides answer."""
        response = "ClearPath is a project management tool with great features."
        flags = evaluator.evaluate(response, chunks_retrieved=0, sources=[])
        assert "no_context" in flags
    
    def test_no_context_with_refusal(self, evaluator):
        """Should NOT flag when chunks=0 but LLM refuses appropriately."""
        response = "I don't have information about that in the documentation."
        flags = evaluator.evaluate(response, chunks_retrieved=0, sources=[])
        assert "no_context" not in flags
        assert "refusal" in flags
    
    def test_no_flag_when_chunks_present(self, evaluator, sample_chunks):
        """Should NOT flag when chunks were retrieved."""
        response = "The Pro plan costs $25/month."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert "no_context" not in flags


class TestRefusalDetection:
    """Tests for refusal detection."""
    
    def test_refusal_phrases(self, evaluator, sample_chunks):
        """Should detect various refusal phrases."""
        refusal_responses = [
            "I don't have information about that feature.",
            "This is not mentioned in the documentation.",
            "I cannot find details about that.",
            "I don't know the answer to that question.",
            "There is no information available about this.",
            "I can't provide details on that.",
            "Unable to find information about this topic."
        ]
        
        for response in refusal_responses:
            flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
            assert "refusal" in flags, f"Failed to detect refusal in: {response}"
    
    def test_no_refusal_in_normal_answer(self, evaluator, sample_chunks):
        """Should NOT flag normal answers as refusals."""
        response = "The Pro plan costs $25/month and includes API access."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert "refusal" not in flags


class TestGroundednessCheck:
    """Tests for unverified feature detection."""
    
    def test_unverified_integration(self, evaluator):
        """Should flag when LLM mentions integration not in chunks."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="ClearPath integrates with Google Calendar and Microsoft Teams.",
                    document_name="integrations.pdf",
                    page_number=1,
                    token_count=10
                ),
                relevance_score=0.9
            )
        ]
        
        # LLM mentions Slack which is not in chunks
        response = "ClearPath integrates with Slack, Google Calendar, and Microsoft Teams."
        flags = evaluator.evaluate(response, chunks_retrieved=1, sources=chunks)
        assert "unverified_feature" in flags
    
    def test_verified_features(self, evaluator):
        """Should NOT flag when all features are in chunks."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="ClearPath offers custom workflows, API access, and priority support in the Pro plan.",
                    document_name="features.pdf",
                    page_number=1,
                    token_count=15
                ),
                relevance_score=0.9
            )
        ]
        
        response = "The Pro plan includes custom workflows and API access."
        flags = evaluator.evaluate(response, chunks_retrieved=1, sources=chunks)
        assert "unverified_feature" not in flags
    
    def test_proper_noun_extraction(self, evaluator):
        """Test proper noun extraction logic."""
        text = "ClearPath integrates with Slack and GitHub for seamless collaboration."
        proper_nouns = evaluator._extract_proper_nouns(text)
        
        # Should extract capitalized proper nouns
        assert "ClearPath" in proper_nouns or "Slack" in proper_nouns or "GitHub" in proper_nouns
    
    def test_no_flag_for_common_words(self, evaluator, sample_chunks):
        """Should NOT flag common capitalized words."""
        response = "The Pro plan is great. This plan includes many features."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        # Should not flag "The", "This", "Pro" (Pro is in chunks)
        assert "unverified_feature" not in flags


class TestPricingUncertainty:
    """Tests for pricing uncertainty detection."""
    
    def test_hedging_language(self, evaluator, sample_chunks):
        """Should flag pricing responses with hedging language."""
        hedging_responses = [
            "The Pro plan may cost around $25/month.",
            "Pricing might vary depending on your needs.",
            "The cost is approximately $25 per month.",
            "Enterprise pricing could be around $100/month.",
            "The fee varies by plan."
        ]
        
        for response in hedging_responses:
            flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
            assert "pricing_uncertainty" in flags, f"Failed to detect hedging in: {response}"
    
    def test_conflicting_pricing_sources(self, evaluator):
        """Should flag when multiple pricing documents retrieved."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="Pro plan costs $25/month.",
                    document_name="pricing_2024.pdf",
                    page_number=1,
                    token_count=5
                ),
                relevance_score=0.9
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc2_1_0",
                    text="Enterprise pricing starts at $100/month.",
                    document_name="enterprise_pricing.pdf",
                    page_number=1,
                    token_count=5
                ),
                relevance_score=0.85
            )
        ]
        
        response = "The Pro plan costs $25/month."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=chunks)
        assert "pricing_uncertainty" in flags
    
    def test_no_flag_for_definitive_pricing(self, evaluator, sample_chunks):
        """Should NOT flag definitive pricing statements."""
        response = "The Pro plan costs $25/month."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert "pricing_uncertainty" not in flags
    
    def test_no_flag_for_non_pricing_queries(self, evaluator, sample_chunks):
        """Should NOT flag non-pricing responses even with hedging."""
        response = "The feature may be available in the next update."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert "pricing_uncertainty" not in flags


class TestMultipleFlags:
    """Tests for scenarios with multiple flags."""
    
    def test_no_context_and_no_refusal(self, evaluator):
        """Can have no_context without refusal."""
        response = "ClearPath is great for teams."
        flags = evaluator.evaluate(response, chunks_retrieved=0, sources=[])
        assert "no_context" in flags
        assert "refusal" not in flags
    
    def test_refusal_and_no_context_mutually_exclusive(self, evaluator):
        """Refusal prevents no_context flag."""
        response = "I don't have information about that."
        flags = evaluator.evaluate(response, chunks_retrieved=0, sources=[])
        assert "refusal" in flags
        assert "no_context" not in flags
    
    def test_multiple_quality_issues(self, evaluator):
        """Can have multiple flags simultaneously."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc1_1_0",
                    text="ClearPath offers various plans.",
                    document_name="pricing_old.pdf",
                    page_number=1,
                    token_count=5
                ),
                relevance_score=0.9
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="doc2_1_0",
                    text="Pricing information for Enterprise.",
                    document_name="pricing_new.pdf",
                    page_number=1,
                    token_count=5
                ),
                relevance_score=0.85
            )
        ]
        
        # Response with unverified feature (Slack) and hedging language
        response = "ClearPath may integrate with Slack and pricing varies by plan."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=chunks)
        
        # Should have both unverified_feature and pricing_uncertainty
        assert "unverified_feature" in flags
        assert "pricing_uncertainty" in flags


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_response(self, evaluator, sample_chunks):
        """Should handle empty response."""
        flags = evaluator.evaluate("", chunks_retrieved=2, sources=sample_chunks)
        # Empty response shouldn't crash, might not trigger any flags
        assert isinstance(flags, list)
    
    def test_very_long_response(self, evaluator, sample_chunks):
        """Should handle very long responses."""
        response = "The Pro plan " * 1000 + "costs $25/month."
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert isinstance(flags, list)
    
    def test_special_characters(self, evaluator, sample_chunks):
        """Should handle special characters in response."""
        response = "The Pro plan costs $25/month! 🎉 It's great!!!"
        flags = evaluator.evaluate(response, chunks_retrieved=2, sources=sample_chunks)
        assert isinstance(flags, list)
