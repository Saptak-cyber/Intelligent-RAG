"""Output evaluator for response quality checks."""
import re
from typing import List, Set
from models.chunk import ScoredChunk

try:
    import spacy
    # Load small English model for named entity recognition
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    nlp = None


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
    ) -> tuple[List[str], dict]:
        """
        Evaluate response quality and return flags with details.
        
        Args:
            response: Generated LLM response
            chunks_retrieved: Number of chunks retrieved
            sources: Retrieved chunks with metadata
            
        Returns:
            Tuple of (flags list, details dict) where details contains additional info
        """
        flags = []
        details = {}
        
        # Check 1: No-context detection
        if self._is_no_context(response, chunks_retrieved):
            flags.append("no_context")
        
        # Check 2: Refusal detection
        if self._is_refusal(response):
            flags.append("refusal")
        
        # Check 3: Groundedness check (unverified features)
        unverified_result = self._has_unverified_features(response, sources)
        if unverified_result["has_unverified"]:
            flags.append("unverified_feature")
            details["unverified_feature"] = {
                "unverified_nouns": sorted(unverified_result["unverified_nouns"]),
                "response_nouns": sorted(unverified_result["response_nouns"]),
                "chunks_nouns": sorted(unverified_result["chunks_nouns"]),
                "spacy_available": SPACY_AVAILABLE
            }
        
        # Check 4: Pricing uncertainty detection
        if self._has_pricing_uncertainty(response, sources):
            flags.append("pricing_uncertainty")
        
        return flags, details
    
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
    ) -> dict:
        """
        Detect when LLM mentions features/integrations not in retrieved chunks.

        This catches hallucinated features based on general SaaS knowledge.
        Uses proper noun extraction to identify specific features mentioned.

        Returns:
            Dict with:
                - has_unverified: bool
                - unverified_nouns: set of unverified nouns
                - response_nouns: set of all nouns in response
                - chunks_nouns: set of all nouns in chunks
        """
        # Extract proper nouns from response (capitalized terms, integration names)
        response_proper_nouns = self._extract_proper_nouns(response)

        if not response_proper_nouns:
            return {
                "has_unverified": False,
                "unverified_nouns": set(),
                "response_nouns": set(),
                "chunks_nouns": set()
            }

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
            return {
                "has_unverified": len(significant_nouns) > 0,
                "unverified_nouns": significant_nouns,
                "response_nouns": response_proper_nouns,
                "chunks_nouns": set()
            }

        # Combine all chunk text and convert to lowercase for safe searching
        chunks_text_lower = " ".join([chunk.chunk.text for chunk in sources]).lower()

        # Extract proper nouns from chunks for comparison
        chunks_text = " ".join([chunk.chunk.text for chunk in sources])
        chunks_proper_nouns = self._extract_proper_nouns(chunks_text)

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
            "your", "my", "our", "their", "his", "her", "its"
        }
        
        # Additional filtering for malformed extractions
        significant_unverified = set()
        for noun in unverified_nouns:
            # Skip if too short
            if len(noun) <= 2:
                continue
            
            # Skip if in stop words
            if noun in stop_words:
                continue
            
            # Skip if contains special characters (except spaces and hyphens)
            if re.search(r'[^a-z0-9\s\-]', noun):
                continue
            
            # Skip if starts with stop words
            first_word = noun.split()[0] if ' ' in noun else noun
            if first_word in stop_words:
                continue
            
            significant_unverified.add(noun)

        return {
            "has_unverified": len(significant_unverified) > 0,
            "unverified_nouns": significant_unverified,
            "response_nouns": response_proper_nouns,
            "chunks_nouns": chunks_proper_nouns
        }
    
    def _extract_proper_nouns(self, text: str) -> Set[str]:
        """
        Extract proper nouns from text using spaCy NER for accurate detection.
        Falls back to pattern matching if spaCy is unavailable.

        Uses spaCy to identify:
        - ORG: Organizations, companies, agencies (e.g., "Slack", "GitHub")
        - PRODUCT: Products, services (e.g., "ClearPath")
        - GPE: Geopolitical entities (e.g., "Paris", "France")
        
        Also includes integration patterns for comprehensive coverage.
        """
        proper_nouns = set()

        if SPACY_AVAILABLE:
            # Use spaCy for accurate named entity recognition
            doc = nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                # Focus on organizations, products, and locations
                if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                    # Clean the entity text
                    cleaned = ent.text.lower().strip()
                    
                    # Filter out malformed extractions with special characters
                    if re.search(r'[^a-z0-9\s\-]', cleaned):
                        continue
                    
                    # Split multi-word entities into individual words
                    # This prevents "the Slack integration" from being treated as one entity
                    words = cleaned.split()
                    
                    # Common stop words to filter out
                    stop_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
                    
                    for word in words:
                        # Skip stop words and short words
                        if word in stop_words or len(word) <= 2:
                            continue
                        
                        # Add individual word as proper noun
                        proper_nouns.add(word)
        else:
            # Fallback: Simple capitalized word extraction
            # This is less accurate but works without spaCy
            words = text.split()
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w\s-]', '', word.replace("'s", "").replace("\u2019s", ""))
                
                if not clean_word or not clean_word[0].isupper():
                    continue
                
                # Only add if clearly a proper noun (CamelCase or all caps)
                if clean_word.isupper() or any(c.isupper() for c in clean_word[1:]):
                    proper_nouns.add(clean_word.lower())

        # Always include integration patterns for comprehensive coverage
        integration_patterns = [
            # Collaboration & Project Management
            r'\b(slack|github|jira|trello|asana|monday|notion|confluence|basecamp|clickup)\b',
            # Major Platforms
            r'\b(google\s+\w+|microsoft\s+\w+|salesforce|adobe\s+\w+)\b',
            # DevOps & Infrastructure
            r'\b(aws|azure|docker|kubernetes|gitlab|bitbucket|jenkins|circleci|heroku|vercel|netlify)\b',
            # Monitoring & Analytics
            r'\b(datadog|sentry|pagerduty|splunk|new\s+relic|grafana|prometheus)\b',
            # Communication & Support
            r'\b(zoom|teams|zendesk|intercom|freshdesk|hubspot)\b',
            # Design & Content
            r'\b(figma|sketch|miro|canva|airtable)\b',
            # Payment & Finance
            r'\b(stripe|paypal|square|quickbooks)\b',
            # Storage & File Sharing
            r'\b(dropbox|box|onedrive|s3)\b',
            # API & Auth protocols
            r'\b(rest\s+api|graphql|oauth|sso|saml|jwt|api\s+key)\b'
        ]

        for pattern in integration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
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
