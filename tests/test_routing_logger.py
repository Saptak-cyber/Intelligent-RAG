"""Unit tests for RoutingLogger."""
import sys
sys.path.insert(0, 'backend')

import json
import pytest
from pathlib import Path
from services.routing_logger import RoutingLogger


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file path."""
    log_file = tmp_path / "test_routing_decisions.jsonl"
    return str(log_file)


@pytest.fixture
def routing_logger(temp_log_file):
    """Create a RoutingLogger instance with temporary log file."""
    logger = RoutingLogger(log_file_path=temp_log_file)
    yield logger
    logger.close()


def test_log_routing_decision_creates_file(routing_logger, temp_log_file):
    """Test that logging creates the log file."""
    routing_logger.log_routing_decision(
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
        }
    )
    
    # Verify file was created
    assert Path(temp_log_file).exists()


def test_log_routing_decision_json_format(routing_logger, temp_log_file):
    """Test that log entries are in JSON Lines format."""
    routing_logger.log_routing_decision(
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
        evaluator_flags=["no_context"]
    )
    
    # Read and parse the log entry
    with open(temp_log_file, 'r') as f:
        log_line = f.readline()
        log_entry = json.loads(log_line)
    
    # Verify all required fields are present
    assert "timestamp" in log_entry
    assert log_entry["query"] == "How do I configure custom workflows?"
    assert log_entry["classification"] == "complex"
    assert log_entry["model_used"] == "llama-3.3-70b-versatile"
    assert log_entry["rule_triggered"] == "complex_keyword"
    assert log_entry["tokens_input"] == 512
    assert log_entry["tokens_output"] == 128
    assert log_entry["latency_ms"] == 687
    assert log_entry["chunks_retrieved"] == 3
    assert log_entry["evaluator_flags"] == ["no_context"]


def test_log_routing_decision_complexity_score(routing_logger, temp_log_file):
    """Test that complexity_score object is logged correctly."""
    complexity_score = {
        "word_count": 15,
        "complex_keyword_count": 2,
        "question_mark_count": 1,
        "comparison_word_count": 1
    }
    
    routing_logger.log_routing_decision(
        query="Compare Enterprise vs Pro features and explain the differences",
        classification="complex",
        model_used="llama-3.3-70b-versatile",
        tokens_input=450,
        tokens_output=200,
        latency_ms=800,
        rule_triggered="comparison_words",
        complexity_score=complexity_score
    )
    
    # Read and parse the log entry
    with open(temp_log_file, 'r') as f:
        log_line = f.readline()
        log_entry = json.loads(log_line)
    
    # Verify complexity_score is present and correct
    assert "complexity_score" in log_entry
    assert log_entry["complexity_score"]["word_count"] == 15
    assert log_entry["complexity_score"]["complex_keyword_count"] == 2
    assert log_entry["complexity_score"]["question_mark_count"] == 1
    assert log_entry["complexity_score"]["comparison_word_count"] == 1


def test_log_routing_decision_rule_triggered(routing_logger, temp_log_file):
    """Test that rule_triggered field is logged correctly."""
    test_cases = [
        ("ood_filter", "Hello!"),
        ("complex_keyword", "Why is this happening?"),
        ("query_length", "This is a very long query with more than fifteen words in it"),
        ("multiple_questions", "What is this? How does it work?"),
        ("comparison_words", "Compare A vs B"),
        ("default", "List features")
    ]
    
    for rule, query in test_cases:
        routing_logger.log_routing_decision(
            query=query,
            classification="simple" if rule in ["ood_filter", "default"] else "complex",
            model_used="llama-3.1-8b-instant",
            tokens_input=100,
            tokens_output=50,
            latency_ms=300,
            rule_triggered=rule,
            complexity_score={
                "word_count": len(query.split()),
                "complex_keyword_count": 0,
                "question_mark_count": query.count("?"),
                "comparison_word_count": 0
            }
        )
    
    # Read all log entries
    with open(temp_log_file, 'r') as f:
        log_entries = [json.loads(line) for line in f]
    
    # Verify all rules were logged
    logged_rules = [entry["rule_triggered"] for entry in log_entries]
    expected_rules = [rule for rule, _ in test_cases]
    assert logged_rules == expected_rules


def test_log_routing_decision_optional_fields(routing_logger, temp_log_file):
    """Test that optional fields have default values."""
    routing_logger.log_routing_decision(
        query="Test query",
        classification="simple",
        model_used="llama-3.1-8b-instant",
        tokens_input=100,
        tokens_output=50,
        latency_ms=300,
        rule_triggered="default",
        complexity_score={
            "word_count": 2,
            "complex_keyword_count": 0,
            "question_mark_count": 0,
            "comparison_word_count": 0
        }
    )
    
    # Read and parse the log entry
    with open(temp_log_file, 'r') as f:
        log_line = f.readline()
        log_entry = json.loads(log_line)
    
    # Verify optional fields have default values
    assert log_entry["chunks_retrieved"] == 0
    assert log_entry["evaluator_flags"] == []
    assert log_entry["system_prompt_tokens"] == 0
    assert log_entry["context_tokens"] == 0


def test_log_routing_decision_multiple_entries(routing_logger, temp_log_file):
    """Test that multiple log entries are written correctly."""
    # Log multiple decisions
    for i in range(5):
        routing_logger.log_routing_decision(
            query=f"Query {i}",
            classification="simple",
            model_used="llama-3.1-8b-instant",
            tokens_input=100 + i,
            tokens_output=50 + i,
            latency_ms=300 + i,
            rule_triggered="default",
            complexity_score={
                "word_count": 2,
                "complex_keyword_count": 0,
                "question_mark_count": 0,
                "comparison_word_count": 0
            }
        )
    
    # Read all log entries
    with open(temp_log_file, 'r') as f:
        log_entries = [json.loads(line) for line in f]
    
    # Verify all entries were written
    assert len(log_entries) == 5
    
    # Verify entries are in order
    for i, entry in enumerate(log_entries):
        assert entry["query"] == f"Query {i}"
        assert entry["tokens_input"] == 100 + i
        assert entry["tokens_output"] == 50 + i
        assert entry["latency_ms"] == 300 + i


def test_log_directory_creation(tmp_path):
    """Test that log directory is created if it doesn't exist."""
    log_file = tmp_path / "nested" / "dir" / "routing_decisions.jsonl"
    logger = RoutingLogger(log_file_path=str(log_file))
    
    logger.log_routing_decision(
        query="Test",
        classification="simple",
        model_used="llama-3.1-8b-instant",
        tokens_input=100,
        tokens_output=50,
        latency_ms=300,
        rule_triggered="default",
        complexity_score={
            "word_count": 1,
            "complex_keyword_count": 0,
            "question_mark_count": 0,
            "comparison_word_count": 0
        }
    )
    
    # Verify directory and file were created
    assert log_file.exists()
    assert log_file.parent.exists()
    
    logger.close()
