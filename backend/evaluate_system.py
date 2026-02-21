"""
Evaluation test harness for ClearPath RAG Chatbot.

This script executes a comprehensive test suite against the chatbot API,
measuring routing accuracy, retrieval quality, latency, token usage, and
evaluator flag frequency.

Usage:
    python evaluate_system.py [--api-url http://localhost:8000] [--output logs/evaluation_report.txt]
"""
import argparse
import json
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
import sys


@dataclass
class TestQuery:
    """Test query with expected behavior."""
    id: int
    question: str
    expected_routing: str  # "simple" or "complex"
    expected_skip_retrieval: bool = False
    expected_flags: List[str] = None
    category: str = ""  # For grouping in report
    
    def __post_init__(self):
        if self.expected_flags is None:
            self.expected_flags = []


@dataclass
class QueryResult:
    """Result from executing a test query."""
    query_id: int
    question: str
    answer: str
    model_used: str
    classification: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: List[str]
    sources: List[Dict[str, Any]]
    error: Optional[str] = None


class EvaluationHarness:
    """Test harness for evaluating the RAG chatbot system."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize evaluation harness.
        
        Args:
            api_url: Base URL for the chatbot API
        """
        self.api_url = api_url
        self.results: List[QueryResult] = []
        
    def load_test_queries(self) -> List[TestQuery]:
        """
        Load test queries from the enhanced test questions file.
        
        Returns:
            List of TestQuery objects
        """
        queries = []
        
        # Part 1: Model Router Test Questions
        
        # Simple Questions (Default Rule) - Questions 1-15
        simple_questions = [
            "What is BambooHR used for at ClearPath?",
            "How much is the learning stipend?",
            "What is the probationary period?",
            "What is Carta used for?",
            "When are salary reviews conducted?",
            "What is the home office stipend amount?",
            "What are the core collaboration hours?",
            "What is the overtime rate?",
            "How often are employees paid?",
            "What is the wellness stipend?",
            "What VPN client should remote employees use?",
            "What is the keyboard shortcut to create a new task?",
            "How many federal holidays does ClearPath observe?",
            "What is the minimum PTO recommendation?",
            "What is the file upload size limit on the Free plan?",
        ]
        
        for i, q in enumerate(simple_questions, start=1):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=False,
                category="Simple Questions"
            ))
        
        # Complex Questions - Keyword Triggered (16-30)
        complex_keyword_questions = [
            "Explain the difference between exempt and non-exempt employees at ClearPath.",
            "Compare the API rate limits between Pro and Enterprise plans.",
            "Analyze the security features included in the Enterprise plan.",
            "What's the difference between the Free plan's 500MB storage and Pro's 50GB?",
            "Explain how the equity vesting schedule works at ClearPath.",
            "Compare the support response times across Free, Pro, and Enterprise plans.",
            "What's the relationship between the probationary period and performance reviews?",
            "Explain how the data retention policy differs by data type.",
            "Compare the mobile app features between iOS and Android.",
            "Analyze the benefits of annual billing versus monthly billing.",
            "What's the difference between the Timeline view and Gantt chart export?",
            "Explain the relationship between story points and velocity reports.",
            "Compare the onboarding process for Pro versus Enterprise customers.",
            "What's the difference between SAML 2.0 and OAuth 2.0 for SSO?",
            "Analyze the migration timeline from Jira to ClearPath.",
        ]
        
        for i, q in enumerate(complex_keyword_questions, start=16):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="complex",
                expected_skip_retrieval=False,
                category="Complex - Keywords"
            ))
        
        # Complex Questions - Length Triggered (31-35)
        complex_length_questions = [
            "Can you walk me through the complete process of setting up a new workspace including inviting team members, creating projects, and configuring integrations?",
            "What are all the steps involved in the Enterprise migration process from discovery through validation and how long does each phase typically take?",
            "I need to understand the full security and compliance certifications that ClearPath Enterprise has including SOC 2, GDPR, CCPA, HIPAA, and ISO 27001 status.",
            "What is the detailed breakdown of the onboarding checklist for new ClearPath customers covering all four weeks including setup, customization, team adoption, and optimization?",
            "Can you explain the entire performance review process at ClearPath including self-assessment, peer feedback, manager review, calibration sessions, and the review conversation timeline?",
        ]
        
        for i, q in enumerate(complex_length_questions, start=31):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="complex",
                expected_skip_retrieval=False,
                category="Complex - Length"
            ))
        
        # Complex Questions - Multiple Questions (36-45)
        complex_multiple_questions = [
            "What is the Pro plan price? How many users does it support? What storage is included?",
            "How do I set up Slack integration? What events trigger notifications? Can I customize them?",
            "What is the equity vesting schedule? When is the cliff? How do I exercise options?",
            "How do I export my data? What formats are available? Are attachments included?",
            "What is the cancellation policy? Do I get a refund? When does access end?",
            "How do I reset my password? What if the email doesn't arrive? How long is the link valid?",
            "What integrations are available? Which plans include them? How do I connect them?",
            "What is the SLA for Enterprise? What response times are guaranteed? Are there credits?",
            "How do I delete my account? Is it reversible? What happens to my data?",
            "What is the API rate limit? How do I check remaining requests? What happens if I exceed it?",
        ]
        
        for i, q in enumerate(complex_multiple_questions, start=36):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="complex",
                expected_skip_retrieval=False,
                category="Complex - Multiple Questions"
            ))
        
        # Complex Questions - Comparison Words (46-55)
        complex_comparison_questions = [
            "Pro versus Enterprise - which plan is better for a 30-person team?",
            "Is annual billing better than monthly for cost savings?",
            "How does ClearPath's Timeline view compare to traditional Gantt charts?",
            "Which is better for security: IP whitelisting or geo-restrictions?",
            "Monthly vs annual contracts - what are the discount differences?",
            "Is the mobile app better on iOS or Android?",
            "How does ClearPath's pricing compare to Jira and Asana?",
            "Which is worse for performance: large projects or many integrations?",
            "Email support versus phone support - what's the difference?",
            "CSV export compared to JSON export - which should I use?",
        ]
        
        for i, q in enumerate(complex_comparison_questions, start=46):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="complex",
                expected_skip_retrieval=False,
                category="Complex - Comparison"
            ))
        
        # OOD Questions (56-70)
        ood_questions = [
            "Hi", "Hello", "Hey", "Thanks", "Thank you",
            "Hello!", "Thanks!", "Thank you so much", "Hey there", "Hi there",
            "Who are you?", "What can you do?", "Help", "Help me", "What is this?"
        ]
        
        for i, q in enumerate(ood_questions, start=56):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=True,
                category="OOD - Should Skip Retrieval"
            ))
        
        # Edge Cases - Should NOT Trigger OOD (71-78)
        edge_not_ood = [
            "Hi, how do I reset my password?",
            "Hello, what is the Pro plan pricing?",
            "Thanks for that, but can you also tell me about the Enterprise plan?",
            "I need help with setting up Slack integration.",
            "Can you help me understand the API rate limits?",
            "What help resources are available?",
            "Who are you supposed to contact for security incidents?",
            "What can you do with the API?",
        ]
        
        for i, q in enumerate(edge_not_ood, start=71):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=False,
                category="Edge - Not OOD"
            ))
        
        # Edge Cases - CSV/VS Bug Test (79-85)
        csv_questions = [
            ("What is the CSV export format?", "simple", False),
            ("How do I export data to CSV?", "simple", False),
            ("Can I import CSV files?", "simple", False),
            ("Does ClearPath support CSV or JSON exports?", "simple", False),
            ("What is the versus operator in SQL?", "simple", False),
            ("Pro vs Enterprise pricing comparison", "complex", False),
            ("Which is better: CSV or JSON?", "complex", False),
        ]
        
        for i, (q, routing, skip) in enumerate(csv_questions, start=79):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing=routing,
                expected_skip_retrieval=skip,
                category="Edge - CSV/VS"
            ))
        
        # Part 2: Output Evaluator Test Questions
        
        # No Context Questions (86-95)
        no_context_questions = [
            "What is the weather in San Francisco today?",
            "Who won the 2023 World Series?",
            "What is the capital of France?",
            "How do I make chocolate chip cookies?",
            "What is quantum entanglement?",
            "Who is the current President of the United States?",
            "What is the stock price of Apple?",
            "How many planets are in the solar system?",
            "What is the meaning of life?",
            "How do I learn Python programming?",
        ]
        
        for i, q in enumerate(no_context_questions, start=86):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=False,
                expected_flags=["no_context"],
                category="Evaluator - No Context"
            ))
        
        # Refusal Questions (96-105)
        refusal_questions = [
            "What is ClearPath's office address in San Francisco?",
            "Who is the CEO of ClearPath?",
            "What is ClearPath's annual revenue?",
            "How many customers does ClearPath have globally?",
            "What is the employee headcount at ClearPath?",
            "What is ClearPath's valuation?",
            "Who are ClearPath's investors?",
            "What is the company's profit margin?",
            "When was ClearPath's last funding round?",
            "What is the churn rate for ClearPath customers?",
        ]
        
        for i, q in enumerate(refusal_questions, start=96):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=False,
                expected_flags=["refusal"],
                category="Evaluator - Refusal"
            ))
        
        # Specific Detail Tests (155-170) - Sample subset
        specific_detail_questions = [
            "What percentage of employee premiums does ClearPath cover for health insurance?",
            "What is the company match for 401(k)?",
            "How many times annual salary is the life insurance coverage?",
            "How much does ClearPath contribute to HSA annually?",
            "What is the maximum home office stipend for remote workers?",
            "How many years is the equity vesting period?",
            "How many days notice is required for PTO exceeding 5 business days?",
            "How many federal holidays does ClearPath observe?",
            "What is the minimum password length requirement?",
            "How often must passwords be changed for confidential data access?",
        ]
        
        for i, q in enumerate(specific_detail_questions, start=155):
            queries.append(TestQuery(
                id=i,
                question=q,
                expected_routing="simple",
                expected_skip_retrieval=False,
                category="Specific Details"
            ))
        
        return queries
    
    def execute_query(self, query: TestQuery) -> QueryResult:
        """
        Execute a single test query against the API.
        
        Args:
            query: TestQuery to execute
            
        Returns:
            QueryResult with response data
        """
        try:
            response = requests.post(
                f"{self.api_url}/query",
                json={"question": query.question},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            return QueryResult(
                query_id=query.id,
                question=query.question,
                answer=data.get("answer", ""),
                model_used=data.get("metadata", {}).get("model_used", ""),
                classification=data.get("metadata", {}).get("classification", ""),
                tokens_input=data.get("metadata", {}).get("tokens", {}).get("input", 0),
                tokens_output=data.get("metadata", {}).get("tokens", {}).get("output", 0),
                latency_ms=data.get("metadata", {}).get("latency_ms", 0),
                chunks_retrieved=data.get("metadata", {}).get("chunks_retrieved", 0),
                evaluator_flags=data.get("metadata", {}).get("evaluator_flags", []),
                sources=data.get("sources", [])
            )
            
        except requests.exceptions.RequestException as e:
            return QueryResult(
                query_id=query.id,
                question=query.question,
                answer="",
                model_used="",
                classification="",
                tokens_input=0,
                tokens_output=0,
                latency_ms=0,
                chunks_retrieved=0,
                evaluator_flags=[],
                sources=[],
                error=str(e)
            )
    
    def run_evaluation(self, queries: List[TestQuery], delay_ms: int = 100) -> None:
        """
        Execute all test queries and collect results.
        
        Args:
            queries: List of TestQuery objects to execute
            delay_ms: Delay between queries in milliseconds
        """
        print(f"Running evaluation with {len(queries)} test queries...")
        print(f"API URL: {self.api_url}")
        print()
        
        for i, query in enumerate(queries, start=1):
            print(f"[{i}/{len(queries)}] Executing query {query.id}: {query.question[:60]}...")
            
            result = self.execute_query(query)
            self.results.append(result)
            
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  ✓ {result.classification} | {result.model_used} | "
                      f"{result.latency_ms}ms | {result.chunks_retrieved} chunks | "
                      f"flags: {result.evaluator_flags}")
            
            # Delay between requests to avoid rate limiting
            if i < len(queries):
                time.sleep(delay_ms / 1000.0)
        
        print()
        print(f"Evaluation complete. Processed {len(self.results)} queries.")
    
    def calculate_metrics(self, queries: List[TestQuery]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from results.
        
        Args:
            queries: Original test queries with expected behavior
            
        Returns:
            Dictionary of metrics
        """
        # Create lookup for expected behavior
        expected = {q.id: q for q in queries}
        
        # Filter out errors
        valid_results = [r for r in self.results if not r.error]
        error_count = len(self.results) - len(valid_results)
        
        # Routing accuracy
        routing_correct = 0
        routing_total = 0
        skip_retrieval_correct = 0
        skip_retrieval_total = 0
        
        for result in valid_results:
            if result.query_id in expected:
                exp = expected[result.query_id]
                
                # Check routing classification
                if result.classification == exp.expected_routing:
                    routing_correct += 1
                routing_total += 1
                
                # Check skip_retrieval behavior
                if exp.expected_skip_retrieval:
                    if result.chunks_retrieved == 0:
                        skip_retrieval_correct += 1
                    skip_retrieval_total += 1
        
        routing_accuracy = routing_correct / routing_total if routing_total > 0 else 0
        skip_retrieval_accuracy = skip_retrieval_correct / skip_retrieval_total if skip_retrieval_total > 0 else 0
        
        # Latency distribution
        latencies = [r.latency_ms for r in valid_results]
        latency_p50 = statistics.median(latencies) if latencies else 0
        latency_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else 0
        latency_p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else 0
        latency_mean = statistics.mean(latencies) if latencies else 0
        
        # Token usage
        total_input_tokens = sum(r.tokens_input for r in valid_results)
        total_output_tokens = sum(r.tokens_output for r in valid_results)
        total_tokens = total_input_tokens + total_output_tokens
        
        # Token usage by query type
        simple_tokens = sum(r.tokens_input + r.tokens_output 
                           for r in valid_results if r.classification == "simple")
        complex_tokens = sum(r.tokens_input + r.tokens_output 
                            for r in valid_results if r.classification == "complex")
        
        # Model usage
        model_counts = Counter(r.model_used for r in valid_results)
        
        # Evaluator flags
        flag_counts = Counter()
        for result in valid_results:
            for flag in result.evaluator_flags:
                flag_counts[flag] += 1
        
        # Retrieval quality
        chunks_retrieved = [r.chunks_retrieved for r in valid_results]
        avg_chunks = statistics.mean(chunks_retrieved) if chunks_retrieved else 0
        queries_with_chunks = sum(1 for c in chunks_retrieved if c > 0)
        
        # Category breakdown
        category_stats = defaultdict(lambda: {"count": 0, "avg_latency": 0, "avg_tokens": 0})
        for result in valid_results:
            if result.query_id in expected:
                cat = expected[result.query_id].category
                category_stats[cat]["count"] += 1
                category_stats[cat]["avg_latency"] += result.latency_ms
                category_stats[cat]["avg_tokens"] += result.tokens_input + result.tokens_output
        
        for cat in category_stats:
            count = category_stats[cat]["count"]
            if count > 0:
                category_stats[cat]["avg_latency"] /= count
                category_stats[cat]["avg_tokens"] /= count
        
        return {
            "total_queries": len(self.results),
            "successful_queries": len(valid_results),
            "failed_queries": error_count,
            "routing": {
                "accuracy": routing_accuracy,
                "correct": routing_correct,
                "total": routing_total,
                "skip_retrieval_accuracy": skip_retrieval_accuracy,
                "skip_retrieval_correct": skip_retrieval_correct,
                "skip_retrieval_total": skip_retrieval_total
            },
            "latency": {
                "mean_ms": latency_mean,
                "p50_ms": latency_p50,
                "p95_ms": latency_p95,
                "p99_ms": latency_p99
            },
            "tokens": {
                "total": total_tokens,
                "input": total_input_tokens,
                "output": total_output_tokens,
                "simple_queries": simple_tokens,
                "complex_queries": complex_tokens
            },
            "models": dict(model_counts),
            "evaluator_flags": dict(flag_counts),
            "retrieval": {
                "avg_chunks_retrieved": avg_chunks,
                "queries_with_chunks": queries_with_chunks,
                "queries_without_chunks": len(valid_results) - queries_with_chunks
            },
            "categories": dict(category_stats)
        }
    
    def generate_report(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Generate evaluation report and save to file.
        
        Args:
            metrics: Calculated metrics dictionary
            output_path: Path to save report
        """
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ClearPath RAG Chatbot - Evaluation Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"API URL: {self.api_url}")
        report_lines.append("")
        
        # Summary
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Queries:      {metrics['total_queries']}")
        report_lines.append(f"Successful:         {metrics['successful_queries']}")
        report_lines.append(f"Failed:             {metrics['failed_queries']}")
        report_lines.append("")
        
        # Routing Performance
        report_lines.append("-" * 80)
        report_lines.append("ROUTING PERFORMANCE")
        report_lines.append("-" * 80)
        routing = metrics['routing']
        report_lines.append(f"Classification Accuracy:  {routing['accuracy']:.2%} "
                          f"({routing['correct']}/{routing['total']})")
        report_lines.append(f"Skip Retrieval Accuracy:  {routing['skip_retrieval_accuracy']:.2%} "
                          f"({routing['skip_retrieval_correct']}/{routing['skip_retrieval_total']})")
        report_lines.append("")
        
        # Model Usage
        report_lines.append("-" * 80)
        report_lines.append("MODEL USAGE")
        report_lines.append("-" * 80)
        for model, count in metrics['models'].items():
            pct = count / metrics['successful_queries'] * 100 if metrics['successful_queries'] > 0 else 0
            report_lines.append(f"{model:40s} {count:5d} queries ({pct:5.1f}%)")
        report_lines.append("")
        
        # Latency Distribution
        report_lines.append("-" * 80)
        report_lines.append("LATENCY DISTRIBUTION")
        report_lines.append("-" * 80)
        latency = metrics['latency']
        report_lines.append(f"Mean:  {latency['mean_ms']:8.1f} ms")
        report_lines.append(f"P50:   {latency['p50_ms']:8.1f} ms")
        report_lines.append(f"P95:   {latency['p95_ms']:8.1f} ms")
        report_lines.append(f"P99:   {latency['p99_ms']:8.1f} ms")
        report_lines.append("")
        
        # Token Usage
        report_lines.append("-" * 80)
        report_lines.append("TOKEN USAGE")
        report_lines.append("-" * 80)
        tokens = metrics['tokens']
        report_lines.append(f"Total Tokens:       {tokens['total']:,}")
        report_lines.append(f"  Input Tokens:     {tokens['input']:,}")
        report_lines.append(f"  Output Tokens:    {tokens['output']:,}")
        report_lines.append(f"Simple Queries:     {tokens['simple_queries']:,} tokens")
        report_lines.append(f"Complex Queries:    {tokens['complex_queries']:,} tokens")
        if metrics['successful_queries'] > 0:
            avg_tokens = tokens['total'] / metrics['successful_queries']
            report_lines.append(f"Avg per Query:      {avg_tokens:.1f} tokens")
        report_lines.append("")
        
        # Retrieval Quality
        report_lines.append("-" * 80)
        report_lines.append("RETRIEVAL QUALITY")
        report_lines.append("-" * 80)
        retrieval = metrics['retrieval']
        report_lines.append(f"Avg Chunks Retrieved:     {retrieval['avg_chunks_retrieved']:.2f}")
        report_lines.append(f"Queries with Chunks:      {retrieval['queries_with_chunks']}")
        report_lines.append(f"Queries without Chunks:   {retrieval['queries_without_chunks']}")
        report_lines.append("")
        
        # Evaluator Flags
        report_lines.append("-" * 80)
        report_lines.append("EVALUATOR FLAGS")
        report_lines.append("-" * 80)
        if metrics['evaluator_flags']:
            for flag, count in sorted(metrics['evaluator_flags'].items(), key=lambda x: -x[1]):
                pct = count / metrics['successful_queries'] * 100 if metrics['successful_queries'] > 0 else 0
                report_lines.append(f"{flag:30s} {count:5d} occurrences ({pct:5.1f}%)")
        else:
            report_lines.append("No evaluator flags raised")
        report_lines.append("")
        
        # Category Breakdown
        report_lines.append("-" * 80)
        report_lines.append("CATEGORY BREAKDOWN")
        report_lines.append("-" * 80)
        for category, stats in sorted(metrics['categories'].items()):
            report_lines.append(f"\n{category}:")
            report_lines.append(f"  Queries:       {stats['count']}")
            report_lines.append(f"  Avg Latency:   {stats['avg_latency']:.1f} ms")
            report_lines.append(f"  Avg Tokens:    {stats['avg_tokens']:.1f}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("End of Report")
        report_lines.append("=" * 80)
        
        # Write to file
        report_text = "\n".join(report_lines)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        # Also print to console
        print(report_text)
        print()
        print(f"Report saved to: {output_path}")


def main():
    """Main entry point for evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Evaluation test harness for ClearPath RAG Chatbot"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL for the chatbot API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output",
        default="logs/evaluation_report.txt",
        help="Output path for evaluation report (default: logs/evaluation_report.txt)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Delay between queries in milliseconds (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Initialize harness
    harness = EvaluationHarness(api_url=args.api_url)
    
    # Load test queries
    print("Loading test queries...")
    queries = harness.load_test_queries()
    print(f"Loaded {len(queries)} test queries")
    print()
    
    # Run evaluation
    harness.run_evaluation(queries, delay_ms=args.delay)
    
    # Calculate metrics
    print()
    print("Calculating metrics...")
    metrics = harness.calculate_metrics(queries)
    
    # Generate report
    print()
    print("Generating report...")
    harness.generate_report(metrics, args.output)
    
    # Exit with error code if there were failures
    if metrics['failed_queries'] > 0:
        print()
        print(f"WARNING: {metrics['failed_queries']} queries failed")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
