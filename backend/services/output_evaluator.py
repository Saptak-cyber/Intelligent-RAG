"""Output evaluator for response quality checks."""
import re
from typing import List, Set
from models.chunk import ScoredChunk


class OutputEvaluator:
    """Analyzes generated responses and flags quality issues."""
    
    # Refusal phrases to detect when LLM declines to answer
    REFUSAL_PHRASES = [
        "i don't have",
        "not mentioned",
        "cannot find",
        "don't know",
        "no information",
        "i cannot",
        "i can't",
        "unable to find",
        "not available",
        "doesn't mention"
    ]
    
    # Hedging language for pricing uncertainty
    HEDGING_PHRASES = [
        "may",
        "might",
        "approximately",
        "around",
        "varies",
        "could be",
        "possibly",
        "perhaps",
        "roughly"
    ]
    
    # Pricing-related keywords
    PRICING_KEYWORDS = [
        "price",
        "pricing",
        "cost",
        "fee",
        "plan",
        "subscription",
        "payment",
        "charge"
    ]
    
    def evaluate(
        self,
        response: str,
        chunks_retrieved: int,
        sources: List[ScoredChunk]
    ) -> List[str]:
        """
        Evaluate response quality and return flags.
        
        Args:
            response: Generated LLM response
            chunks_retrieved: Number of chunks retrieved
            sources: Retrieved chunks with metadata
            
        Returns:
            List of flag strings (empty if no issues)
        """
        flags = []
        
        # Check 1: No-context detection
        if self._is_no_context(response, chunks_retrieved):
            flags.append("no_context")
        
        # Check 2: Refusal detection
        if self._is_refusal(response):
            flags.append("refusal")
        
        # Check 3: Groundedness check (unverified features)
        if self._has_unverified_features(response, sources):
            flags.append("unverified_feature")
        
        # Check 4: Pricing uncertainty detection
        if self._has_pricing_uncertainty(response, sources):
            flags.append("pricing_uncertainty")
        
        return flags
    
    def _is_no_context(self, response: str, chunks_retrieved: int) -> bool:
        """
        Detect when LLM answers without documentation support.
        
        Condition: chunks_retrieved == 0 AND response is not a refusal
        """
        if chunks_retrieved > 0:
            return False
        
        # If no chunks retrieved but LLM refused to answer, that's appropriate
        if self._is_refusal(response):
            return False
        
        # LLM generated an answer without any context - potential hallucination
        return True
    
    def _is_refusal(self, response: str) -> bool:
        """
        Detect when LLM explicitly refuses to answer.
        
        Checks for refusal phrases in the response.
        """
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.REFUSAL_PHRASES)
    
    def _has_unverified_features(
        self,
        response: str,
        sources: List[ScoredChunk]
    ) -> bool:
        """
        Detect when LLM mentions features/integrations not in retrieved chunks.
        
        This catches hallucinated features based on general SaaS knowledge.
        Uses proper noun extraction to identify specific features mentioned.
        """
        # Extract proper nouns from response (capitalized terms, integration names)
        response_proper_nouns = self._extract_proper_nouns(response)
        
        if not response_proper_nouns:
            return False
        
        # Extract proper nouns from all retrieved chunks
        chunks_text = " ".join([chunk.chunk.text for chunk in sources])
        chunks_proper_nouns = self._extract_proper_nouns(chunks_text)
        
        # Check if response mentions proper nouns not in chunks
        unverified_nouns = response_proper_nouns - chunks_proper_nouns
        
        # Filter out common words that might be capitalized but aren't features
        # (e.g., "The", "I", "A", single letters, very short words)
        significant_unverified = {
            noun for noun in unverified_nouns
            if len(noun) > 2 and noun not in {"The", "This", "That", "These", "Those"}
        }
        
        return len(significant_unverified) > 0
    
    def _extract_proper_nouns(self, text: str) -> Set[str]:
        """
        Extract proper nouns from text using regex patterns.
        
        Looks for:
        - Capitalized words (potential product names, features)
        - Integration names (e.g., "Slack", "GitHub")
        - Multi-word proper nouns (e.g., "Pro Plan")
        """
        proper_nouns = set()
        
        # Pattern 1: Capitalized words (but not at sentence start)
        # Look for words that are capitalized in the middle of sentences
        words = text.split()
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)
            
            # Skip if empty after cleaning
            if not clean_word:
                continue
            
            # Check if word is capitalized
            if clean_word[0].isupper():
                # Skip if it's the first word after sentence-ending punctuation
                if i > 0 and not words[i-1].rstrip().endswith(('.', '!', '?', ':')):
                    proper_nouns.add(clean_word)
                # Also add if it's clearly a proper noun (all caps or mixed case)
                elif clean_word.isupper() or (clean_word[0].isupper() and any(c.isupper() for c in clean_word[1:])):
                    proper_nouns.add(clean_word)
        
        # Pattern 2: Common integration/tool names (case-insensitive search, case-sensitive storage)
        integration_patterns = [
            r'\b(Slack|GitHub|Jira|Trello|Asana|Monday|Notion|Confluence)\b',
            r'\b(Google|Microsoft|Apple|Amazon|Salesforce)\b',
            r'\b(API|REST|GraphQL|OAuth|SSO|SAML)\b'
        ]
        
        for pattern in integration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Store with original capitalization from text
                proper_nouns.add(match.group(0))
        
        return proper_nouns
    
    def _has_pricing_uncertainty(
        self,
        response: str,
        sources: List[ScoredChunk]
    ) -> bool:
        """
        Detect pricing-related responses with uncertainty or conflicting sources.
        
        Condition: Query mentions pricing AND (hedging language OR conflicting sources)
        """
        response_lower = response.lower()
        
        # Check if response is about pricing
        is_pricing_related = any(
            keyword in response_lower for keyword in self.PRICING_KEYWORDS
        )
        
        if not is_pricing_related:
            return False
        
        # Check for hedging language
        has_hedging = any(
            phrase in response_lower for phrase in self.HEDGING_PHRASES
        )
        
        if has_hedging:
            return True
        
        # Check for conflicting sources (multiple different documents about pricing)
        if len(sources) >= 2:
            # Get unique document names
            doc_names = set(chunk.chunk.document_name for chunk in sources)
            
            # If multiple different pricing-related documents, might be conflicting
            pricing_docs = [
                doc for doc in doc_names
                if any(keyword in doc.lower() for keyword in self.PRICING_KEYWORDS)
            ]
            
            if len(pricing_docs) >= 2:
                return True
        
        return False
