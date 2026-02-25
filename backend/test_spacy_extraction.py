"""
Test spaCy-based proper noun extraction for unverified_feature flag.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.output_evaluator import OutputEvaluator, SPACY_AVAILABLE
from models.chunk import Chunk, ScoredChunk

print("=" * 80)
print("TESTING SPACY-BASED PROPER NOUN EXTRACTION")
print("=" * 80)

print(f"\nspaCy Available: {SPACY_AVAILABLE}")

if not SPACY_AVAILABLE:
    print("\n⚠️  WARNING: spaCy not available. Install with:")
    print("   pip install spacy")
    print("   python -m spacy download en_core_web_sm")
    print("\nFalling back to simple extraction...")

# Test response with common false positives
response = """Slack Integration with ClearPath

You're interested with learning more about our Slack integration with ClearPath. We're happy to help.

What's New in Slack Integration v2?

We've recently introduced Slack integration v2, which brings improved functionality and seamless communication between ClearPath and Slack. This integration allows you to:

• Send notifications: Receive notifications in ClearPath when someone mentions you or assigns a task to you in Slack.
• Share updates: Share updates from ClearPath to Slack, keeping your team informed about project progress.
• Integrate workflows: Integrate your ClearPath workflows with Slack channels, enabling team members to access project data directly from Slack.

How to Set Up Slack Integration

To set up Slack integration, follow these steps:

1. Go to Settings in ClearPath.
2. Click on Integrations.
3. Search for Slack in the integrations list.
4. Click on the Slack icon to connect your Slack workspace.
5. Authorize ClearPath to access your Slack workspace.
6. Configure the integration settings to suit your needs."""

# Chunks that contain Slack
chunk1_text = """What's New in Slack Integration v2?
We've recently introduced Slack integration v2, which brings improved functionality and seamless communication between ClearPath and Slack."""

chunk2_text = """How do I integrate ClearPath with Slack?
To set up Slack integration, follow these steps:
1. Go to Settings in ClearPath.
2. Click on Integrations."""

# Create scored chunks
chunk1 = Chunk(
    chunk_id="test_1",
    text=chunk1_text,
    document_name="30_Release_Notes_Version_History.pdf",
    page_number=2,
    embedding=[0.1] * 1536
)

chunk2 = Chunk(
    chunk_id="test_2",
    text=chunk2_text,
    document_name="17_FAQ_Common_Questions.pdf",
    page_number=1,
    embedding=[0.1] * 1536
)

scored_chunks = [
    ScoredChunk(chunk=chunk1, relevance_score=0.565),
    ScoredChunk(chunk=chunk2, relevance_score=0.402)
]

# Test the evaluator
evaluator = OutputEvaluator()

print("\n" + "-" * 80)
print("TEST 1: Extract proper nouns from RESPONSE")
print("-" * 80)

response_nouns = evaluator._extract_proper_nouns(response)
print(f"\nExtracted {len(response_nouns)} proper nouns:")
for noun in sorted(response_nouns):
    print(f"  • {noun}")

print("\n" + "-" * 80)
print("TEST 2: Extract proper nouns from CHUNKS")
print("-" * 80)

chunks_text = " ".join([chunk.chunk.text for chunk in scored_chunks])
chunks_nouns = evaluator._extract_proper_nouns(chunks_text)
print(f"\nExtracted {len(chunks_nouns)} proper nouns:")
for noun in sorted(chunks_nouns):
    print(f"  • {noun}")

print("\n" + "-" * 80)
print("TEST 3: Check for unverified features")
print("-" * 80)

has_unverified = evaluator._has_unverified_features(response, scored_chunks)
print(f"\nHas unverified features: {has_unverified}")

if has_unverified:
    unverified = response_nouns - chunks_nouns
    stop_words = {
        "the", "this", "that", "these", "those", "it", "they", "we", "you",
        "a", "an", "and", "or", "but", "for", "in", "on", "at", "to", "of", "with",
        "your", "my", "our", "their", "his", "her", "its"
    }
    significant = {n for n in unverified if len(n) > 2 and n not in stop_words}
    print(f"\nUnverified nouns: {sorted(significant)}")

print("\n" + "-" * 80)
print("TEST 4: Full evaluation")
print("-" * 80)

flags = evaluator.evaluate(response, chunks_retrieved=2, sources=scored_chunks)
print(f"\nEvaluation flags: {flags}")

print("\n" + "-" * 80)
print("EXPECTED BEHAVIOR")
print("-" * 80)
print("✓ Should extract: 'slack', 'clearpath' (actual proper nouns)")
print("✗ Should NOT extract: 'set', 'go', 'click', 'integrations' (common words)")
print("✓ Should NOT flag unverified_feature (Slack is in chunks)")

print("\n" + "=" * 80)
