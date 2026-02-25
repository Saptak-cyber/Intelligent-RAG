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
        "doesn't mention",
        "does not contain",
        "does not mention",
        "do not have",
        "does not have"
    ]
    
    # Hedging language for pricing uncertainty
    HEDGING_PHRASES = [
        "might",
        "approximately",
        "could be",
        "possibly",
        "perhaps",
        "roughly"
    ]
    
    # Strong pricing-related keywords (high confidence)
    STRONG_PRICING_KEYWORDS = [
        "price",
        "pricing",
        "cost",
        "fee",
        "subscription"
    ]
    
    # Weak pricing-related keywords (need additional context)
    WEAK_PRICING_KEYWORDS = [
        "plan",
        "payment",
        "charge",
        "revenue",
        "income",
        "earnings",
        "financial"
    ]
    
    # Phrases indicating conflicting or unclear documentation
    CONFLICT_PHRASES = [
        "conflict",
        "contradict",
        "contradictory",
        "different prices",
        "inconsistent",
        "discrepancy",
        "unclear",
        "not explicitly stated",
        "multiple prices listed",
        "differing information"
    ]
    
    # Indicators that the model is providing a partial answer rather than a total refusal
    PARTIAL_ANSWER_INDICATORS = [
        "but",
        "however",
        "although",
        "on the other hand",
        "does mention",
        "is available",
        "instead",
        "alternatively"
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
        Detect when LLM explicitly refuses to answer the entirety of the question.
        Avoids flagging partial answers where the LLM provides some valid information.
        """
        response_lower = response.lower()
        
        # 1. Check for refusal phrases using regex word boundaries
        has_refusal = False
        for phrase in self.REFUSAL_PHRASES:
            if re.search(rf'\b{re.escape(phrase)}\b', response_lower):
                has_refusal = True
                break
        
        if not has_refusal:
            return False
        
        # 2. Check if response provides substantive answer content
        # These phrases indicate the LLM is providing actual information
        provides_answer_indicators = [
            "does mention",
            "is available",
            "according to the",
            "clearpath offers",
            "clearpath provides",
            "clearpath supports",
            "you can use",
            "it includes",
            "features include",
            "the price is",
            "the cost is",
            "plans include"
        ]
        
        provides_substantive_answer = any(
            indicator in response_lower 
            for indicator in provides_answer_indicators
        )
        
        # If refusal phrase found but also provides substantive content, it's a partial answer
        return not provides_substantive_answer
    
    # def _has_unverified_features(
    #     self,
    #     response: str,
    #     sources: List[ScoredChunk]
    # ) -> bool:
    #     """
    #     Detect when LLM mentions features/integrations not in retrieved chunks.
        
    #     This catches hallucinated features based on general SaaS knowledge.
    #     Uses proper noun extraction to identify specific features mentioned.
    #     """
    #     # Extract proper nouns from response (capitalized terms, integration names)
    #     response_proper_nouns = self._extract_proper_nouns(response)
        
    #     if not response_proper_nouns:
    #         return False
        
    #     # Extract proper nouns from all retrieved chunks
    #     chunks_text = " ".join([chunk.chunk.text for chunk in sources])
    #     chunks_proper_nouns = self._extract_proper_nouns(chunks_text)
        
    #     # Check if response mentions proper nouns not in chunks
    #     unverified_nouns = response_proper_nouns - chunks_proper_nouns
        
    #     # Filter out common words that might be capitalized but aren't features
    #     # Note: These are now lowercase to match the updated extractor logic
    #     stop_words = {
    #         "the", "this", "that", "these", "those", "it", "they", "we", "you",
    #         "a", "an", "and", "or", "but", "for"
    #     }
        
    #     significant_unverified = {
    #         noun for noun in unverified_nouns
    #         if len(noun) > 2 and noun not in stop_words
    #     }
        
    #     return len(significant_unverified) > 0
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
        
        # Handle empty sources list
        if not sources:
            # If no sources but response has proper nouns, they're unverified
            stop_words = {
                "the", "this", "that", "these", "those", "it", "they", "we", "you",
                "a", "an", "and", "or", "but", "for", "in", "on", "at", "to", "of", "with"
            }
            significant_nouns = {
                noun for noun in response_proper_nouns
                if len(noun) > 2 and noun not in stop_words
            }
            return len(significant_nouns) > 0
        
        # Combine all chunk text and convert to lowercase for safe searching
        chunks_text_lower = " ".join([chunk.chunk.text for chunk in sources]).lower()
        
        # Check if the extracted nouns exist ANYWHERE in the source chunks
        unverified_nouns = set()
        for noun in response_proper_nouns:
            # Use regex boundaries to ensure we match whole words only
            if not re.search(rf'\b{re.escape(noun)}\b', chunks_text_lower):
                unverified_nouns.add(noun)
        
        # Filter out common words that might be capitalized but aren't features
        stop_words = {
            "the", "this", "that", "these", "those", "it", "they", "we", "you",
            "a", "an", "and", "or", "but", "for", "in", "on", "at", "to", "of", "with",
            "your", "my", "our", "their", "his", "her", "its","set", "up", "integrations", "youre", "go", "click", "search", "configure"
        }
        
        significant_unverified = {
            noun for noun in unverified_nouns
            if len(noun) > 2 and noun not in stop_words
        }
        
        return len(significant_unverified) > 0
    
    def _extract_proper_nouns(self, text: str) -> Set[str]:
        """
        Extract proper nouns from text using regex patterns, handling Markdown
        and normalizing casing to prevent false positives.

        Looks for:
        - Capitalized words (potential product names, features)
        - Integration names (e.g., "Slack", "GitHub")

        Excludes common UI terms and generic capitalized words.
        """
        proper_nouns = set()

        # Common UI/generic terms that shouldn't be flagged (lowercase for comparison)
        ui_terms = {
            'email', 'profile', 'settings', 'account', 'password', 'username',
            'dashboard', 'menu', 'button', 'page', 'tab', 'field', 'form',
            'update', 'save', 'cancel', 'delete', 'edit', 'view', 'create',
            'address', 'name', 'phone', 'date', 'time', 'status', 'type',
            'notes', 'description', 'title', 'message', 'notification',
            'hours', 'days', 'weeks', 'months', 'years', 'new', 'old'
        }

        # Pattern 1: Capitalized words (but not at sentence start)
        # Look for words that are capitalized in the middle of sentences
        words = text.split()
        for i, word in enumerate(words):
            # Strip possessives first to avoid "ClearPath's" becoming "ClearPaths"
            word = word.replace("'s", "").replace("\u2019s", "")

            # Clean remaining punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)

            # Skip if empty after cleaning
            if not clean_word:
                continue

            # Check if word is capitalized
            if clean_word[0].isupper():
                is_sentence_start = (i == 0)

                # Check previous word for sentence terminators or markdown markers
                if i > 0:
                    prev_word = words[i-1].strip()
                    # Matches punctuation endings OR markdown list markers (e.g., -, *, +, >, •, 1., 1))
                    if (prev_word.endswith(('.', '!', '?', ':')) or 
                        re.match(r'^(\d+[.)]|[-*+>•])$', prev_word)):
                        is_sentence_start = True

                word_lower = clean_word.lower()

                # Skip common UI terms
                if word_lower in ui_terms:
                    continue

                # Add if it's mid-sentence and not a UI term
                if not is_sentence_start:
                    proper_nouns.add(word_lower)
                # Add if it's clearly a proper noun (all caps or camelCase) even at start of sentence
                elif clean_word.isupper() or any(c.isupper() for c in clean_word[1:]):
                    proper_nouns.add(word_lower)

        # Pattern 2: Common integration/tool names (case-insensitive search, lowercase storage)
        # Removed overly generic terms like "api" to reduce false positives
        integration_patterns = [
            r'\b(slack|github|jira|trello|asana|monday|notion|confluence)\b',
            r'\b(google\s+\w+|microsoft\s+\w+|salesforce)\b',  # Require context
            r'\b(rest\s+api|graphql|oauth|sso|saml)\b'  # More specific API terms
        ]

        for pattern in integration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Store as lowercase to ensure case-insensitive set difference
                proper_nouns.add(match.group(0).lower())

        return proper_nouns
    
    def _has_pricing_uncertainty(
        self,
        response: str,
        sources: List[ScoredChunk]
    ) -> bool:
        """
        Detect pricing-related responses that express uncertainty or flag conflicting sources.
        
        Condition: Response mentions pricing AND (uses hedging language OR explicitly mentions conflicts OR is a refusal)
        """
        response_lower = response.lower()
        
        # Check if response is about pricing using word boundaries
        # Strong keywords alone are sufficient
        has_strong_keyword = any(
            re.search(rf'\b{re.escape(keyword)}\b', response_lower)
            for keyword in self.STRONG_PRICING_KEYWORDS
        )
        
        # Weak keywords need at least 2 to confirm pricing context
        weak_keyword_count = sum(
            1 for keyword in self.WEAK_PRICING_KEYWORDS
            if re.search(rf'\b{re.escape(keyword)}\b', response_lower)
        )
        
        is_pricing_related = has_strong_keyword or weak_keyword_count >= 2
        
        if not is_pricing_related:
            return False
        
        # Check for hedging language using word boundaries
        has_hedging = any(
            re.search(rf'\b{re.escape(phrase)}\b', response_lower)
            for phrase in self.HEDGING_PHRASES
        )
        
        if has_hedging:
            return True
        
        # Check for explicit mention of conflicting or unclear documentation
        has_conflict = any(
            phrase in response_lower for phrase in self.CONFLICT_PHRASES
        )
        
        if has_conflict:
            return True
        
        # Check if this is a refusal to a pricing-related question
        # This catches cases where the system can't find pricing/revenue information
        if self._is_refusal(response):
            return True
        
        return False
