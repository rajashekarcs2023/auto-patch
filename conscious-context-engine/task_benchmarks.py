"""
Concrete Task Performance Benchmarks
Real-world tasks that demonstrate reasoning improvement
"""
import asyncio
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from reasoning_engine import MetaReasoningEngine, ReasoningTaskEnvironment
import art
from openai import AsyncOpenAI
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class TaskBenchmark:
    """Concrete task with measurable performance metrics"""
    task_id: str
    task_type: str
    problem: str
    success_criteria: List[str]
    expected_output_keywords: List[str]
    difficulty_level: float
    time_limit_seconds: int = 300


class TaskPerformanceTracker:
    """Track task performance improvements"""
    
    def __init__(self):
        self.task_benchmarks = self._create_concrete_benchmarks()
        self.performance_history = []
        self.baseline_established = False
        
    def _create_concrete_benchmarks(self) -> List[TaskBenchmark]:
        """Create concrete, measurable tasks"""
        return [
            # Code Architecture Task
            TaskBenchmark(
                task_id="microservices_oauth",
                task_type="technical",
                problem="Design a complete OAuth2 implementation for a microservices architecture with 20 services. Include security flows, token management, service-to-service authentication, and rate limiting. Provide implementation details.",
                success_criteria=["security_design", "scalability_plan", "implementation_steps", "error_handling"],
                expected_output_keywords=["jwt", "client_credentials", "authorization_code", "refresh_token", "rate_limiting", "circuit_breaker", "service_mesh"],
                difficulty_level=0.9,
                time_limit_seconds=240
            ),
            
            # System Diagnosis Task  
            TaskBenchmark(
                task_id="performance_debugging",
                task_type="diagnosis",
                problem="A distributed system handling 100K requests/second suddenly drops to 5K requests/second. CPU usage is normal, memory usage is 60%, network latency increased 10x. Database connections are at 80% capacity. Diagnose the root cause and provide solution steps.",
                success_criteria=["systematic_analysis", "root_cause_identification", "solution_steps", "prevention_measures"],
                expected_output_keywords=["connection_pooling", "database_locks", "query_optimization", "indexing", "connection_timeout", "deadlocks"],
                difficulty_level=0.85,
                time_limit_seconds=180
            ),
            
            # Algorithm Design Task
            TaskBenchmark(
                task_id="real_time_matching",
                task_type="algorithmic",
                problem="Design a real-time matching algorithm for a ride-sharing service that matches 10,000 concurrent ride requests with available drivers within 2 seconds, optimizing for both distance and estimated time of arrival. Handle dynamic pricing and driver preferences.",
                success_criteria=["algorithm_efficiency", "real_time_constraints", "optimization_strategy", "edge_case_handling"],
                expected_output_keywords=["geospatial_indexing", "priority_queue", "dynamic_programming", "graph_algorithms", "load_balancing", "cache_strategy"],
                difficulty_level=0.95,
                time_limit_seconds=300
            ),
            
            # Security Analysis Task
            TaskBenchmark(
                task_id="api_security_audit",
                task_type="security",
                problem="Conduct a comprehensive security audit of a REST API that handles financial transactions. The API has 50 endpoints, processes $1M daily, and integrates with 3rd party payment processors. Identify vulnerabilities and provide remediation plan.",
                success_criteria=["vulnerability_identification", "risk_assessment", "remediation_plan", "compliance_check"],
                expected_output_keywords=["sql_injection", "xss", "csrf", "authentication", "authorization", "encryption", "pci_compliance", "rate_limiting"],
                difficulty_level=0.9,
                time_limit_seconds=270
            ),
            
            # Data Architecture Task
            TaskBenchmark(
                task_id="real_time_analytics",
                task_type="data",
                problem="Design a real-time analytics pipeline that processes 1TB of user event data daily, provides sub-second query responses for 1000 concurrent users, and supports both OLTP and OLAP workloads. Include data modeling, storage, and query optimization.",
                success_criteria=["data_modeling", "performance_optimization", "scalability_design", "technology_selection"],
                expected_output_keywords=["stream_processing", "columnar_storage", "partitioning", "indexing", "caching", "materialized_views", "lambda_architecture"],
                difficulty_level=0.9,
                time_limit_seconds=240
            )
        ]
    
    async def benchmark_task_performance(self, model: art.Model, reasoning_engine: MetaReasoningEngine, task: TaskBenchmark) -> Dict[str, Any]:
        """Benchmark performance on a specific task"""
        
        print(f"\nBenchmarking Task: {task.task_id}")
        print(f"Difficulty: {task.difficulty_level}")
        print("-" * 50)
        
        # Get reasoning chain for this task type
        reasoning_chain = reasoning_engine.get_reasoning_chain(task.task_type)
        reasoning_template = reasoning_chain.to_prompt_template()
        
        # Create enhanced system prompt
        system_prompt = f"""You are an expert {task.task_type} specialist. Use this systematic reasoning approach:

{reasoning_template}

For each step, be specific, detailed, and provide concrete implementation details where applicable.
Focus on practical, real-world solutions that can be implemented."""

        user_prompt = f"""Task: {task.problem}

Please provide a comprehensive solution following the reasoning framework. Be thorough and specific."""

        # Time the response
        start_time = datetime.now()
        
        try:
            client = AsyncOpenAI(
                base_url=model.inference_base_url,
                api_key=model.inference_api_key,
            )
            
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2500
            )
            
            response_content = response.choices[0].message.content
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate task performance
            task_scores = self._evaluate_task_performance(response_content, task)
            
            # Record reasoning outcome
            reasoning_score, step_results = self._evaluate_reasoning_steps(response_content, task)
            reasoning_engine.record_reasoning_outcome(
                chain_id=reasoning_chain.chain_id,
                step_results=step_results,
                overall_success=reasoning_score > 0.7,
                task_type=task.task_type
            )
            
            # Check for evolution
            evolved_chain = reasoning_engine.evolve_reasoning_chain(
                reasoning_chain.chain_id,
                task.task_type
            )
            
            result = {
                "task_id": task.task_id,
                "reasoning_chain_id": reasoning_chain.chain_id,
                "reasoning_chain_performance": reasoning_chain.overall_success_rate,
                "task_scores": task_scores,
                "reasoning_score": reasoning_score,
                "response_time": response_time,
                "evolution_triggered": evolved_chain is not None,
                "evolved_chain_id": evolved_chain.chain_id if evolved_chain else None,
                "response_content": response_content,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Task Score: {task_scores['composite']:.3f}")
            print(f"Reasoning Score: {reasoning_score:.3f}")
            print(f"Chain Performance: {reasoning_chain.overall_success_rate:.3f}")
            print(f"Response Time: {response_time:.1f}s")
            if evolved_chain:
                print(f"EVOLUTION: {reasoning_chain.chain_id} -> {evolved_chain.chain_id}")
                
            return result
            
        except Exception as e:
            print(f"Task benchmark failed: {e}")
            return {
                "task_id": task.task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _evaluate_task_performance(self, response: str, task: TaskBenchmark) -> Dict[str, float]:
        """Evaluate task-specific performance metrics"""
        
        scores = {}
        
        # 1. Success Criteria Coverage
        criteria_score = 0
        for criterion in task.success_criteria:
            criterion_words = criterion.replace("_", " ").split()
            if any(word in response.lower() for word in criterion_words):
                criteria_score += 1
        scores["criteria_coverage"] = criteria_score / len(task.success_criteria)
        
        # 2. Expected Keywords Present
        keyword_score = 0
        for keyword in task.expected_output_keywords:
            if keyword.replace("_", " ") in response.lower():
                keyword_score += 1
        scores["keyword_coverage"] = keyword_score / len(task.expected_output_keywords)
        
        # 3. Technical Depth
        depth_indicators = [
            "implementation", "architecture", "design", "algorithm", "optimization",
            "performance", "scalability", "security", "monitoring", "testing"
        ]
        depth_score = sum(1 for indicator in depth_indicators if indicator in response.lower())
        scores["technical_depth"] = min(depth_score / 5, 1.0)
        
        # 4. Solution Completeness
        completeness_indicators = [
            "step", "phase", "approach", "strategy", "solution", "implementation",
            "requirements", "considerations", "challenges", "benefits"
        ]
        completeness_score = sum(1 for indicator in completeness_indicators if indicator in response.lower())
        scores["completeness"] = min(completeness_score / 7, 1.0)
        
        # 5. Response Quality (length and structure)
        quality_score = 0
        if len(response) > 500:  # Detailed response
            quality_score += 0.3
        if len(response.split('\n')) > 5:  # Well structured
            quality_score += 0.3
        if any(marker in response for marker in ['1.', '2.', '•', '-']):  # Organized
            quality_score += 0.4
        scores["response_quality"] = quality_score
        
        # Composite score with task-specific weighting
        weights = {
            "criteria_coverage": 0.3,
            "keyword_coverage": 0.25,
            "technical_depth": 0.2,
            "completeness": 0.15,
            "response_quality": 0.1
        }
        
        scores["composite"] = sum(scores[key] * weights[key] for key in weights)
        
        return scores
    
    def _evaluate_reasoning_steps(self, response: str, task: TaskBenchmark) -> Tuple[float, List[Tuple[int, bool]]]:
        """Evaluate reasoning step quality"""
        
        # Extract reasoning steps
        step_indicators = ["analysis:", "hypothesis:", "verification:", "conclusion:", "approach:", "solution:"]
        step_results = []
        
        for i, indicator in enumerate(step_indicators):
            if indicator in response.lower():
                # Find content after indicator
                start_idx = response.lower().find(indicator)
                if start_idx != -1:
                    # Get next 200 characters to evaluate step quality
                    step_content = response[start_idx:start_idx+200]
                    step_quality = len(step_content) > 50 and any(
                        word in step_content.lower() 
                        for word in task.expected_output_keywords[:3]
                    )
                    step_results.append((i, step_quality))
        
        # Overall reasoning score
        if step_results:
            reasoning_score = sum(1 for _, success in step_results if success) / len(step_results)
        else:
            reasoning_score = 0.3  # Minimal score if no clear reasoning structure
            
        return reasoning_score, step_results
    
    async def run_comprehensive_benchmark(self, model: art.Model, reasoning_engine: MetaReasoningEngine) -> Dict[str, Any]:
        """Run comprehensive benchmark across all tasks"""
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TASK PERFORMANCE BENCHMARK")
        print("="*60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model.name,
            "task_results": {},
            "summary": {}
        }
        
        # Run all benchmarks
        for task in self.task_benchmarks:
            task_result = await self.benchmark_task_performance(model, reasoning_engine, task)
            results["task_results"][task.task_id] = task_result
            
            # Add to performance history
            self.performance_history.append(task_result)
        
        # Calculate summary statistics
        valid_results = [r for r in results["task_results"].values() if "task_scores" in r]
        
        if valid_results:
            composite_scores = [r["task_scores"]["composite"] for r in valid_results]
            reasoning_scores = [r["reasoning_score"] for r in valid_results]
            response_times = [r["response_time"] for r in valid_results]
            evolutions_triggered = sum(1 for r in valid_results if r.get("evolution_triggered", False))
            
            results["summary"] = {
                "total_tasks": len(self.task_benchmarks),
                "successful_tasks": len(valid_results),
                "average_task_score": sum(composite_scores) / len(composite_scores),
                "average_reasoning_score": sum(reasoning_scores) / len(reasoning_scores),
                "average_response_time": sum(response_times) / len(response_times),
                "evolutions_triggered": evolutions_triggered,
                "evolution_rate": evolutions_triggered / len(valid_results),
                "high_performance_tasks": sum(1 for score in composite_scores if score > 0.8),
                "performance_distribution": {
                    "excellent": sum(1 for score in composite_scores if score > 0.9),
                    "good": sum(1 for score in composite_scores if 0.7 < score <= 0.9),
                    "fair": sum(1 for score in composite_scores if 0.5 < score <= 0.7),
                    "poor": sum(1 for score in composite_scores if score <= 0.5)
                }
            }
            
            print(f"\n" + "="*40)
            print("BENCHMARK SUMMARY")
            print("="*40)
            print(f"Average Task Score: {results['summary']['average_task_score']:.3f}")
            print(f"Average Reasoning Score: {results['summary']['average_reasoning_score']:.3f}")
            print(f"Evolutions Triggered: {evolutions_triggered}")
            print(f"High Performance Tasks: {results['summary']['high_performance_tasks']}/{len(valid_results)}")
            print(f"Average Response Time: {results['summary']['average_response_time']:.1f}s")
        
        return results
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"task_benchmark_results_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nBenchmark results saved to: {filename}")
        return filename
    
    def compare_benchmark_runs(self, baseline_file: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current benchmark with baseline"""
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            print(f"Baseline file {baseline_file} not found")
            return {}
        
        comparison = {
            "baseline_timestamp": baseline_results.get("timestamp"),
            "current_timestamp": current_results.get("timestamp"),
            "improvements": {},
            "summary": {}
        }
        
        # Compare individual tasks
        for task_id in current_results["task_results"]:
            if (task_id in baseline_results["task_results"] and 
                "task_scores" in baseline_results["task_results"][task_id] and
                "task_scores" in current_results["task_results"][task_id]):
                
                baseline_score = baseline_results["task_results"][task_id]["task_scores"]["composite"]
                current_score = current_results["task_results"][task_id]["task_scores"]["composite"]
                improvement = current_score - baseline_score
                
                comparison["improvements"][task_id] = {
                    "baseline_score": baseline_score,
                    "current_score": current_score,
                    "improvement": improvement,
                    "improvement_percent": (improvement / baseline_score * 100) if baseline_score > 0 else 0
                }
        
        # Summary comparison
        if baseline_results.get("summary") and current_results.get("summary"):
            baseline_avg = baseline_results["summary"]["average_task_score"]
            current_avg = current_results["summary"]["average_task_score"]
            
            comparison["summary"] = {
                "baseline_average": baseline_avg,
                "current_average": current_avg,
                "overall_improvement": current_avg - baseline_avg,
                "improvement_percent": ((current_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0,
                "tasks_improved": sum(1 for imp in comparison["improvements"].values() if imp["improvement"] > 0),
                "significant_improvements": sum(1 for imp in comparison["improvements"].values() if imp["improvement"] > 0.1)
            }
            
            print(f"\n" + "="*50)
            print("IMPROVEMENT ANALYSIS")
            print("="*50)
            print(f"Overall Improvement: {comparison['summary']['improvement_percent']:.1f}%")
            print(f"Tasks Improved: {comparison['summary']['tasks_improved']}/{len(comparison['improvements'])}")
            print(f"Significant Improvements: {comparison['summary']['significant_improvements']}")
        
        return comparison


async def main():
    """Test the task benchmark system"""
    tracker = TaskPerformanceTracker()
    
    # Mock model for testing
    class MockModel:
        def __init__(self):
            self.name = "test-model"
            self.inference_base_url = "https://api.openai.com/v1"
            self.inference_api_key = os.getenv("OPENAI_API_KEY")
        
        def get_inference_name(self):
            return "gpt-4"
    
    model = MockModel()
    reasoning_engine = MetaReasoningEngine()
    
    # Run single task test
    if tracker.task_benchmarks:
        print("Testing task benchmark system...")
        result = await tracker.benchmark_task_performance(
            model, reasoning_engine, tracker.task_benchmarks[0]
        )
        print(f"\nTest completed. Task score: {result.get('task_scores', {}).get('composite', 0):.3f}")


if __name__ == "__main__":
    asyncio.run(main())