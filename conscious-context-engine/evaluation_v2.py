"""
Advanced Evaluation System for Self-Improving Reasoning
Comprehensive benchmarking and performance measurement
"""
import os
import asyncio
from typing import List, Dict, Tuple, Any
import art
import json
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from litellm import acompletion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ReasoningBenchmark:
    """Comprehensive reasoning benchmark"""
    task_id: str
    task_type: str
    problem_statement: str
    ground_truth: str
    difficulty: float  # 0.0 to 1.0
    evaluation_criteria: List[str]
    expected_reasoning_steps: int


class AdvancedReasoningEvaluator:
    """Advanced evaluation system for reasoning quality"""
    
    def __init__(self):
        self.benchmarks = self._create_comprehensive_benchmarks()
        self.baseline_results = {}
        self.evaluation_history = []
    
    def _create_comprehensive_benchmarks(self) -> List[ReasoningBenchmark]:
        """Create comprehensive reasoning benchmarks"""
        return [
            # Mathematical Reasoning
            ReasoningBenchmark(
                task_id="math_optimization",
                task_type="mathematical",
                problem_statement="A company needs to optimize delivery routes for 50 trucks across 200 locations to minimize total distance while ensuring each location is visited exactly once per day. Design an algorithmic approach.",
                ground_truth="Multi-phase approach: 1) Cluster locations geographically, 2) Apply TSP heuristics within clusters, 3) Optimize inter-cluster routing, 4) Use metaheuristics for global optimization",
                difficulty=0.9,
                evaluation_criteria=["algorithmic_correctness", "scalability_analysis", "optimization_strategy", "implementation_feasibility"],
                expected_reasoning_steps=6
            ),
            
            # System Architecture
            ReasoningBenchmark(
                task_id="distributed_system",
                task_type="technical",
                problem_statement="Design a real-time chat system supporting 10 million concurrent users with 99.99% uptime, end-to-end encryption, and global low-latency delivery.",
                ground_truth="Microservices architecture with: WebSocket gateways, message brokers (Kafka), distributed databases (Cassandra), CDN for static content, load balancers, circuit breakers, and geographic sharding",
                difficulty=0.95,
                evaluation_criteria=["scalability_design", "reliability_measures", "security_implementation", "latency_optimization"],
                expected_reasoning_steps=8
            ),
            
            # Research Analysis
            ReasoningBenchmark(
                task_id="ai_safety_analysis",
                task_type="research",
                problem_statement="Analyze the potential risks and mitigation strategies for deploying large language models in critical infrastructure systems, considering alignment, robustness, and interpretability challenges.",
                ground_truth="Risk assessment covering: alignment drift, adversarial inputs, hallucinations, bias amplification, and failure modes. Mitigation: constitutional AI, robustness testing, interpretability tools, human oversight",
                difficulty=0.85,
                evaluation_criteria=["risk_identification", "mitigation_strategies", "technical_depth", "practical_feasibility"],
                expected_reasoning_steps=7
            ),
            
            # Creative Problem Solving
            ReasoningBenchmark(
                task_id="creative_solution",
                task_type="creative",
                problem_statement="Design a sustainable urban transportation system for a city of 5 million people that reduces carbon emissions by 80% while maintaining accessibility and economic viability.",
                ground_truth="Integrated approach: Electric autonomous vehicle fleets, smart public transit with dynamic routing, cycling infrastructure, pedestrian zones, congestion pricing, and renewable energy integration",
                difficulty=0.8,
                evaluation_criteria=["innovation", "sustainability_impact", "feasibility", "systemic_thinking"],
                expected_reasoning_steps=6
            ),
            
            # Logic and Reasoning
            ReasoningBenchmark(
                task_id="logical_reasoning",
                task_type="logical",
                problem_statement="In a network of trust relationships, if A trusts B, B trusts C, but C doesn't trust A, and trust is not transitive, design a protocol for secure multi-party computation among these parties.",
                ground_truth="Zero-knowledge proof protocol with: commitment schemes, verification without revelation, cryptographic primitives for non-transitive trust, and secure aggregation methods",
                difficulty=0.9,
                evaluation_criteria=["logical_consistency", "security_analysis", "protocol_design", "edge_case_handling"],
                expected_reasoning_steps=5
            )
        ]
    
    async def evaluate_reasoning_comprehensive(self, reasoning_output: str, benchmark: ReasoningBenchmark) -> Dict[str, float]:
        """Comprehensive evaluation of reasoning quality"""
        
        # Multi-faceted evaluation
        scores = {}
        
        # 1. Structural Analysis
        scores["structure"] = self._evaluate_reasoning_structure(reasoning_output, benchmark)
        
        # 2. Content Quality
        scores["content"] = await self._evaluate_content_quality(reasoning_output, benchmark)
        
        # 3. Logical Coherence
        scores["logic"] = self._evaluate_logical_coherence(reasoning_output)
        
        # 4. Completeness
        scores["completeness"] = self._evaluate_completeness(reasoning_output, benchmark)
        
        # 5. Innovation/Creativity
        scores["innovation"] = self._evaluate_innovation(reasoning_output, benchmark)
        
        # Weighted composite score
        weights = {
            "structure": 0.2,
            "content": 0.3,
            "logic": 0.25,
            "completeness": 0.15,
            "innovation": 0.1
        }
        
        composite_score = sum(scores[key] * weights[key] for key in scores)
        scores["composite"] = composite_score
        
        return scores
    
    def _evaluate_reasoning_structure(self, output: str, benchmark: ReasoningBenchmark) -> float:
        """Evaluate the structure and organization of reasoning"""
        
        # Look for clear reasoning steps
        step_indicators = [
            "analysis:", "hypothesis:", "verification:", "conclusion:",
            "step 1", "step 2", "first,", "second,", "then,", "finally,",
            "approach:", "strategy:", "implementation:", "evaluation:"
        ]
        
        steps_found = sum(1 for indicator in step_indicators if indicator.lower() in output.lower())
        structure_score = min(steps_found / benchmark.expected_reasoning_steps, 1.0)
        
        # Bonus for logical flow
        flow_indicators = ["therefore", "because", "thus", "consequently", "as a result"]
        flow_score = min(sum(1 for indicator in flow_indicators if indicator in output.lower()) / 3, 1.0)
        
        return (structure_score * 0.7) + (flow_score * 0.3)
    
    async def _evaluate_content_quality(self, output: str, benchmark: ReasoningBenchmark) -> float:
        """Evaluate content quality using LLM judge"""
        
        evaluation_prompt = f"""
        Evaluate the quality of this reasoning response for the given problem:
        
        Problem: {benchmark.problem_statement}
        Expected elements: {benchmark.ground_truth}
        
        Response to evaluate: {output}
        
        Rate the response on these criteria (0.0 to 1.0 each):
        1. Technical accuracy
        2. Depth of analysis
        3. Practical feasibility
        4. Clarity of explanation
        
        Provide scores as: accuracy=X.X, depth=X.X, feasibility=X.X, clarity=X.X
        """
        
        try:
            response = await acompletion(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of reasoning quality."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Extract scores
            scores = {}
            for criterion in ["accuracy", "depth", "feasibility", "clarity"]:
                try:
                    score_text = content.split(f"{criterion}=")[1].split(",")[0].split()[0]
                    scores[criterion] = float(score_text)
                except:
                    scores[criterion] = 0.5  # Default
            
            return sum(scores.values()) / len(scores)
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return self._fallback_content_evaluation(output, benchmark)
    
    def _fallback_content_evaluation(self, output: str, benchmark: ReasoningBenchmark) -> float:
        """Fallback content evaluation"""
        
        # Check for domain-specific keywords
        domain_keywords = {
            "technical": ["architecture", "system", "scalability", "performance", "implementation"],
            "mathematical": ["algorithm", "optimization", "complexity", "solution", "approach"],
            "research": ["analysis", "methodology", "evidence", "framework", "evaluation"],
            "creative": ["innovation", "design", "sustainable", "integrated", "novel"],
            "logical": ["protocol", "security", "verification", "consistency", "proof"]
        }
        
        keywords = domain_keywords.get(benchmark.task_type, [])
        keyword_score = sum(1 for keyword in keywords if keyword in output.lower()) / len(keywords)
        
        # Length and detail score
        length_score = min(len(output) / 1000, 1.0)  # Normalize to 1000 chars
        
        return (keyword_score * 0.7) + (length_score * 0.3)
    
    def _evaluate_logical_coherence(self, output: str) -> float:
        """Evaluate logical coherence and flow"""
        
        # Check for logical connectors
        logical_connectors = [
            "therefore", "because", "since", "given that", "thus", "hence",
            "as a result", "consequently", "due to", "leads to", "implies"
        ]
        
        connector_count = sum(1 for connector in logical_connectors if connector in output.lower())
        connector_score = min(connector_count / 5, 1.0)
        
        # Check for contradictions (simple heuristic)
        contradiction_indicators = ["but", "however", "although", "despite", "nevertheless"]
        contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in output.lower())
        
        # Too many contradictions might indicate poor coherence
        coherence_penalty = min(contradiction_count * 0.1, 0.3)
        
        return max(connector_score - coherence_penalty, 0.0)
    
    def _evaluate_completeness(self, output: str, benchmark: ReasoningBenchmark) -> float:
        """Evaluate completeness against evaluation criteria"""
        
        criteria_coverage = 0
        for criterion in benchmark.evaluation_criteria:
            # Simple keyword matching (could be improved with semantic similarity)
            criterion_words = criterion.replace("_", " ").split()
            if any(word in output.lower() for word in criterion_words):
                criteria_coverage += 1
        
        return criteria_coverage / len(benchmark.evaluation_criteria)
    
    def _evaluate_innovation(self, output: str, benchmark: ReasoningBenchmark) -> float:
        """Evaluate innovation and creative thinking"""
        
        innovation_indicators = [
            "novel", "innovative", "creative", "unique", "breakthrough",
            "new approach", "alternative", "unconventional", "paradigm",
            "revolutionary", "cutting-edge", "state-of-the-art"
        ]
        
        innovation_count = sum(1 for indicator in innovation_indicators if indicator in output.lower())
        return min(innovation_count / 3, 1.0)


class ModelComparisonBenchmark:
    """Benchmark our model against baselines"""
    
    def __init__(self):
        self.evaluator = AdvancedReasoningEvaluator()
        self.comparison_results = {}
    
    async def benchmark_model_evolution(self, model_before: art.Model, model_after: art.Model, 
                                      reasoning_chain_before: str, reasoning_chain_after: str) -> Dict[str, Any]:
        """Comprehensive benchmark showing model evolution"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {}
        }
        
        for benchmark in self.evaluator.benchmarks:
            print(f"Benchmarking: {benchmark.task_id}")
            
            # Test baseline model
            baseline_response = await self._get_model_response(
                model_before, benchmark.problem_statement, reasoning_chain_before
            )
            baseline_scores = await self.evaluator.evaluate_reasoning_comprehensive(
                baseline_response, benchmark
            )
            
            # Test evolved model
            evolved_response = await self._get_model_response(
                model_after, benchmark.problem_statement, reasoning_chain_after
            )
            evolved_scores = await self.evaluator.evaluate_reasoning_comprehensive(
                evolved_response, benchmark
            )
            
            # Calculate improvement
            improvement = {
                key: evolved_scores[key] - baseline_scores[key] 
                for key in baseline_scores if key in evolved_scores
            }
            
            results["benchmarks"][benchmark.task_id] = {
                "baseline": baseline_scores,
                "evolved": evolved_scores,
                "improvement": improvement,
                "baseline_response": baseline_response,
                "evolved_response": evolved_response
            }
            
            print(f"  Improvement: {improvement['composite']:.3f}")
        
        # Calculate summary metrics
        all_improvements = [
            results["benchmarks"][task]["improvement"]["composite"] 
            for task in results["benchmarks"]
        ]
        
        results["summary"] = {
            "average_improvement": np.mean(all_improvements),
            "median_improvement": np.median(all_improvements),
            "max_improvement": max(all_improvements),
            "min_improvement": min(all_improvements),
            "improvement_std": np.std(all_improvements),
            "tasks_improved": sum(1 for imp in all_improvements if imp > 0),
            "total_tasks": len(all_improvements)
        }
        
        return results
    
    async def _get_model_response(self, model: art.Model, problem: str, reasoning_chain: str) -> str:
        """Get model response for a given problem"""
        
        system_prompt = f"""You are an expert problem solver. Use this reasoning approach:

{reasoning_chain}

Be thorough, systematic, and provide detailed analysis."""

        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=model.inference_base_url,
                api_key=model.inference_api_key,
            )
            
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Problem: {problem}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting model response: {e}")
            return f"Error: Could not get response from model"
    
    def save_benchmark_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {filename}")


# Quick test function
async def test_evaluation_system():
    """Test the evaluation system"""
    evaluator = AdvancedReasoningEvaluator()
    
    # Test response
    test_response = """
    Analysis: This is a complex distributed systems problem requiring careful consideration of scalability, reliability, and performance.
    
    Approach: I will design a microservices architecture with the following components:
    1. Load balancers for traffic distribution
    2. WebSocket gateways for real-time communication
    3. Message brokers for reliable message delivery
    4. Distributed databases for data persistence
    5. Caching layers for performance optimization
    
    Implementation: The system will use horizontal scaling, geographic distribution, and fault tolerance mechanisms.
    
    Verification: This approach addresses the core requirements of high availability, low latency, and scalability.
    """
    
    benchmark = evaluator.benchmarks[1]  # Distributed system benchmark
    scores = await evaluator.evaluate_reasoning_comprehensive(test_response, benchmark)
    
    print("Evaluation Test Results:")
    for key, score in scores.items():
        print(f"  {key}: {score:.3f}")
    
    return scores


if __name__ == "__main__":
    asyncio.run(test_evaluation_system())