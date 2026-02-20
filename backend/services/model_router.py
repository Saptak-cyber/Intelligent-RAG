"""
Model Router for ClearPath RAG Chatbot.

This module implements deterministic query classification using a tiered decision tree
to route queries to appropriate LLM models (llama-3.1-8b-instant or llama-3.3-70b-versatile).
"""

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Classification:
    """
    Result of query classification.
    
    Attributes:
        category: Either "simple" or "complex"
        model_name: The selected LLM model name
        reasoning: Explanation of the classification decision
        skip_retrieval: Whether to skip document retrieval (for OOD queries)
        rule_triggered: Which decision tree rule was applied
    """
    category: str
    model_name: str
    reasoning: str
    skip_retrieval: bool = False
    rule_triggered: str = ""


class ModelRouter:
    """
    Deterministic query classifier that routes queries to appropriate LLM models.
    
    Uses a tiered decision tree with an OOD (Out-of-Distribution) filter to classify
    queries as either "simple" or "complex" based on explicit signals like query length,
    keyword presence, and complexity heuristics.
    """
    
    # Model names
    SIMPLE_MODEL = "llama-3.1-8b-instant"
    COMPLEX_MODEL = "llama-3.3-70b-versatile"
    
    # Classification categories
    SIMPLE = "simple"
    COMPLEX = "complex"
    
    # Keywords for complexity detection
    COMPLEX_KEYWORDS = {
        "why", "how", "explain", "compare", "analyze", "difference", "relationship"
    }
    
    COMPARISON_WORDS = {
        "versus", "vs", "better", "worse", "compared to"
    }
    
    # OOD (Out-of-Distribution) patterns
    GREETING_PATTERNS = {
        "hi", "hello", "hey", "thanks", "thank you"
    }
    
    META_COMMENT_PATTERNS = {
        "who are you", "what can you do", "help"
    }
    
    def classify_query(self, query: str) -> Classification:
        """
        Classify query as simple or complex using deterministic decision tree.
        
        The decision tree uses the following rules in order:
        0. OOD Filter: Greetings or meta-comments → Simple + skip_retrieval
        1. Complex Keywords: Contains complex keywords → Complex
        2. Query Length: > 15 words → Complex
        3. Multiple Questions: > 1 question mark → Complex
        4. Comparison Words: Contains comparison words → Complex
        5. Default: → Simple
        
        Args:
            query: User question string
            
        Returns:
            Classification with category, model name, reasoning, and flags
        """
        if not query or not query.strip():
            logger.warning("Empty query received, classifying as simple")
            return Classification(
                category=self.SIMPLE,
                model_name=self.SIMPLE_MODEL,
                reasoning="Empty query defaults to simple model",
                skip_retrieval=False,
                rule_triggered="default"
            )
        
        query_lower = query.lower().strip()
        
        # Rule 0: OOD Filter - Greetings and meta-comments
        if self._is_greeting(query_lower) or self._is_meta_comment(query_lower):
            reasoning = "Query is a greeting or meta-comment (OOD filter)"
            logger.info(f"Classification: {self.SIMPLE} (OOD filter) - {query[:50]}")
            return Classification(
                category=self.SIMPLE,
                model_name=self.SIMPLE_MODEL,
                reasoning=reasoning,
                skip_retrieval=True,  # Skip retrieval for OOD queries
                rule_triggered="ood_filter"
            )
        
        # Rule 1: Complex Keywords
        if self._contains_complex_keywords(query_lower):
            reasoning = f"Query contains complex keywords: {self._get_matched_keywords(query_lower, self.COMPLEX_KEYWORDS)}"
            logger.info(f"Classification: {self.COMPLEX} (complex keywords) - {query[:50]}")
            return Classification(
                category=self.COMPLEX,
                model_name=self.COMPLEX_MODEL,
                reasoning=reasoning,
                skip_retrieval=False,
                rule_triggered="complex_keyword"
            )
        
        # Rule 2: Query Length
        word_count = len(query.split())
        if word_count > 15:
            reasoning = f"Query length ({word_count} words) exceeds 15 words"
            logger.info(f"Classification: {self.COMPLEX} (query length) - {query[:50]}")
            return Classification(
                category=self.COMPLEX,
                model_name=self.COMPLEX_MODEL,
                reasoning=reasoning,
                skip_retrieval=False,
                rule_triggered="query_length"
            )
        
        # Rule 3: Multiple Questions
        question_mark_count = query.count('?')
        if question_mark_count > 1:
            reasoning = f"Query contains multiple question marks ({question_mark_count})"
            logger.info(f"Classification: {self.COMPLEX} (multiple questions) - {query[:50]}")
            return Classification(
                category=self.COMPLEX,
                model_name=self.COMPLEX_MODEL,
                reasoning=reasoning,
                skip_retrieval=False,
                rule_triggered="multiple_questions"
            )
        
        # Rule 4: Comparison Words
        if self._contains_comparison_words(query_lower):
            reasoning = f"Query contains comparison words: {self._get_matched_keywords(query_lower, self.COMPARISON_WORDS)}"
            logger.info(f"Classification: {self.COMPLEX} (comparison words) - {query[:50]}")
            return Classification(
                category=self.COMPLEX,
                model_name=self.COMPLEX_MODEL,
                reasoning=reasoning,
                skip_retrieval=False,
                rule_triggered="comparison_words"
            )
        
        # Rule 5: Default - Simple
        reasoning = "Query does not match any complexity triggers, defaults to simple"
        logger.info(f"Classification: {self.SIMPLE} (default) - {query[:50]}")
        return Classification(
            category=self.SIMPLE,
            model_name=self.SIMPLE_MODEL,
            reasoning=reasoning,
            skip_retrieval=False,
            rule_triggered="default"
        )
    
    def _is_greeting(self, query_lower: str) -> bool:
        """Check if query is a greeting."""
        # Remove punctuation for greeting detection
        query_clean = re.sub(r'[^\w\s]', '', query_lower).strip()
        
        # Check for exact matches or greetings at the start of the query
        for greeting in self.GREETING_PATTERNS:
            if query_clean == greeting or query_clean.startswith(greeting + " "):
                return True
        return False
    
    def _is_meta_comment(self, query_lower: str) -> bool:
        """Check if query is a meta-comment about the system."""
        for pattern in self.META_COMMENT_PATTERNS:
            if pattern in query_lower:
                return True
        return False
    
    def _contains_complex_keywords(self, query_lower: str) -> bool:
        """Check if query contains any complex keywords."""
        # Use word boundaries to avoid partial matches
        words = set(re.findall(r'\b\w+\b', query_lower))
        return bool(words & self.COMPLEX_KEYWORDS)
    
    def _contains_comparison_words(self, query_lower: str) -> bool:
        """Check if query contains any comparison words."""
        for word in self.COMPARISON_WORDS:
            if word in query_lower:
                return True
        return False
    
    def _get_matched_keywords(self, query_lower: str, keyword_set: set) -> str:
        """Get comma-separated list of matched keywords."""
        words = set(re.findall(r'\b\w+\b', query_lower))
        matched = words & keyword_set
        
        # Also check for multi-word patterns in COMPARISON_WORDS
        for word in keyword_set:
            if ' ' in word and word in query_lower:
                matched.add(word)
        
        return ', '.join(sorted(matched)) if matched else 'none'
