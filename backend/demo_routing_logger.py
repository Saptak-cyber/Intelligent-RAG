"""Demo script for RoutingLogger."""
import sys
import importlib.util
import json

# Import RoutingLogger directly without triggering __init__.py
spec = importlib.util.spec_from_file_location('routing_logger', 'services/routing_logger.py')
routing_logger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing_logger_module)
RoutingLogger = routing_logger_module.RoutingLogger

def main():
    """Demonstrate RoutingLogger usage."""
    print("RoutingLogger Demo")
    print("=" * 50)
    
    # Create logger instance
    logger = RoutingLogger('logs/routing_decisions.jsonl')
    
    # Example 1: Simple query with default rule
    print("\n1. Logging simple query (default rule)...")
    logger.log_routing_decision(
        query="What is the Pro plan price?",
        classification="simple",
        model_used="llama-3.1-8b-instant",
        tokens_input=234,
        tokens_output=45,
        latency_ms=342,
        rule_triggered="default",
        complexity_score={
            "word_count": 6,
            "complex_keyword_count": 0,
            "question_mark_count": 1,
            "comparison_word_count": 0
        },
        chunks_retrieved=2,
        evaluator_flags=[],
        system_prompt_tokens=150,
        context_tokens=84
    )
    
    # Example 2: Complex query with complex keyword rule
    print("2. Logging complex query (complex_keyword rule)...")
    logger.log_routing_decision(
        query="How do I configure custom workflows?",
        classification="complex",
        model_used="llama-3.3-70b-versatile",
        tokens_input=512,
        tokens_output=128,
        latency_ms=687,
        rule_triggered="complex_keyword",
        complexity_score={
            "word_count": 6,
            "complex_keyword_count": 1,
            "question_mark_count": 1,
            "comparison_word_count": 0
        },
        chunks_retrieved=3,
        evaluator_flags=["no_context"],
        system_prompt_tokens=150,
        context_tokens=362
    )
    
    # Example 3: Comparison query
    print("3. Logging comparison query (comparison_words rule)...")
    logger.log_routing_decision(
        query="Compare Enterprise vs Pro features",
        classification="complex",
        model_used="llama-3.3-70b-versatile",
        tokens_input=450,
        tokens_output=200,
        latency_ms=800,
        rule_triggered="comparison_words",
        complexity_score={
            "word_count": 5,
            "complex_keyword_count": 0,
            "question_mark_count": 0,
            "comparison_word_count": 2
        },
        chunks_retrieved=5,
        evaluator_flags=["pricing_uncertainty"],
        system_prompt_tokens=150,
        context_tokens=300
    )
    
    # Example 4: OOD filter (greeting)
    print("4. Logging OOD query (ood_filter rule)...")
    logger.log_routing_decision(
        query="Hello!",
        classification="simple",
        model_used="llama-3.1-8b-instant",
        tokens_input=50,
        tokens_output=20,
        latency_ms=150,
        rule_triggered="ood_filter",
        complexity_score={
            "word_count": 1,
            "complex_keyword_count": 0,
            "question_mark_count": 0,
            "comparison_word_count": 0
        },
        chunks_retrieved=0,
        evaluator_flags=[],
        system_prompt_tokens=50,
        context_tokens=0
    )
    
    logger.close()
    
    # Read and display the log entries
    print("\n" + "=" * 50)
    print("Log entries written to logs/routing_decisions.jsonl:")
    print("=" * 50)
    
    with open('logs/routing_decisions.jsonl', 'r') as f:
        for i, line in enumerate(f, 1):
            log_entry = json.loads(line)
            print(f"\nEntry {i}:")
            print(f"  Query: {log_entry['query']}")
            print(f"  Classification: {log_entry['classification']}")
            print(f"  Model: {log_entry['model_used']}")
            print(f"  Rule: {log_entry['rule_triggered']}")
            print(f"  Complexity Score: {log_entry['complexity_score']}")
            print(f"  Tokens: {log_entry['tokens_input']} in / {log_entry['tokens_output']} out")
            print(f"  Latency: {log_entry['latency_ms']}ms")
            print(f"  Chunks Retrieved: {log_entry['chunks_retrieved']}")
            print(f"  Evaluator Flags: {log_entry['evaluator_flags']}")
    
    print("\n" + "=" * 50)
    print("Demo complete!")

if __name__ == "__main__":
    main()
