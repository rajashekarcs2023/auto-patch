"""
Self-Improving Reasoning Chain Engine
Revolutionary AI that learns to rewrite its own reasoning process
"""
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel
import re


@dataclass
class ReasoningStep:
    """Individual step in a reasoning chain"""
    step_type: str  # "analysis", "hypothesis", "verification", "conclusion"
    content: str
    confidence: float = 0.0
    success_rate: float = 0.0
    usage_count: int = 0


@dataclass 
class ReasoningChain:
    """Complete reasoning pattern for a problem type"""
    chain_id: str
    problem_type: str
    steps: List[ReasoningStep]
    overall_success_rate: float = 0.0
    improvement_count: int = 0
    
    def to_prompt_template(self) -> str:
        """Convert reasoning chain to prompt template"""
        template = f"For {self.problem_type} problems, reason as follows:\n\n"
        for i, step in enumerate(self.steps, 1):
            template += f"{i}. {step.step_type.upper()}: {step.content}\n"
        return template
    
    def get_chain_signature(self) -> str:
        """Get unique signature for this reasoning pattern"""
        content = "".join([step.content for step in self.steps])
        return hashlib.md5(content.encode()).hexdigest()[:8]


class MetaReasoningEngine:
    """Learns which reasoning patterns work best"""
    
    def __init__(self):
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.step_effectiveness: Dict[str, Dict[str, float]] = {}  # step_type -> problem_type -> effectiveness
        self.pattern_library = self._initialize_base_patterns()
        self.improvement_history: List[Dict] = []
        
    def _initialize_base_patterns(self) -> Dict[str, ReasoningChain]:
        """Initialize with basic reasoning patterns"""
        patterns = {}
        
        # Technical problem solving pattern
        tech_steps = [
            ReasoningStep("analysis", "Break down the technical requirements and constraints"),
            ReasoningStep("hypothesis", "Identify the most likely solution approach"),
            ReasoningStep("verification", "Test the approach against known cases"),
            ReasoningStep("conclusion", "Provide concrete implementation steps")
        ]
        patterns["technical"] = ReasoningChain("tech_001", "technical", tech_steps)
        
        # Research pattern
        research_steps = [
            ReasoningStep("analysis", "Identify key information needs and sources"),
            ReasoningStep("hypothesis", "Form initial understanding from available data"),
            ReasoningStep("verification", "Cross-reference with authoritative sources"),
            ReasoningStep("conclusion", "Synthesize findings with confidence levels")
        ]
        patterns["research"] = ReasoningChain("research_001", "research", research_steps)
        
        # Problem diagnosis pattern
        diagnosis_steps = [
            ReasoningStep("analysis", "Gather symptoms and environmental factors"),
            ReasoningStep("hypothesis", "Generate ranked list of potential causes"),
            ReasoningStep("verification", "Test hypotheses systematically"),
            ReasoningStep("conclusion", "Identify root cause and solution path")
        ]
        patterns["diagnosis"] = ReasoningChain("diag_001", "diagnosis", diagnosis_steps)
        
        return patterns
    
    def get_reasoning_chain(self, problem_type: str) -> ReasoningChain:
        """Get best reasoning chain for problem type"""
        # Find best chain for this problem type
        candidates = [chain for chain in self.reasoning_chains.values() 
                     if chain.problem_type == problem_type]
        
        if candidates:
            # Return highest performing chain (consider both success rate and usage)
            best_chain = max(candidates, key=lambda x: (x.overall_success_rate, x.improvement_count))
        else:
            # Use base pattern and immediately add to active chains
            best_chain = self.pattern_library.get(problem_type, 
                                                list(self.pattern_library.values())[0])
            # Create a copy for active use
            import copy
            active_chain = copy.deepcopy(best_chain)
            active_chain.chain_id = f"{problem_type}_active_{len(self.reasoning_chains)}"
            self.reasoning_chains[active_chain.chain_id] = active_chain
            best_chain = active_chain
        
        return best_chain
    
    def record_reasoning_outcome(self, chain_id: str, step_results: List[Tuple[int, bool]], 
                                overall_success: bool, task_type: str) -> None:
        """Record how well a reasoning chain performed"""
        if chain_id not in self.reasoning_chains:
            return
            
        chain = self.reasoning_chains[chain_id]
        
        # Update individual step success rates
        for step_idx, step_success in step_results:
            if step_idx < len(chain.steps):
                step = chain.steps[step_idx]
                step.usage_count += 1
                step.success_rate = ((step.success_rate * (step.usage_count - 1)) + 
                                   (1.0 if step_success else 0.0)) / step.usage_count
        
        # Update overall chain performance
        prev_count = chain.improvement_count
        chain.improvement_count += 1
        chain.overall_success_rate = ((chain.overall_success_rate * prev_count) + 
                                    (1.0 if overall_success else 0.0)) / chain.improvement_count
        
        # Update step effectiveness by problem type
        for step in chain.steps:
            if step.step_type not in self.step_effectiveness:
                self.step_effectiveness[step.step_type] = {}
            if task_type not in self.step_effectiveness[step.step_type]:
                self.step_effectiveness[step.step_type][task_type] = 0.0
            
            # Update effectiveness
            current = self.step_effectiveness[step.step_type][task_type]
            self.step_effectiveness[step.step_type][task_type] = (current + step.success_rate) / 2
    
    def evolve_reasoning_chain(self, chain_id: str, problem_type: str) -> Optional[ReasoningChain]:
        """Create improved version of reasoning chain"""
        if chain_id not in self.reasoning_chains:
            return None
            
        original_chain = self.reasoning_chains[chain_id]
        
        # Only evolve if we have enough data and performance issues
        if original_chain.improvement_count < 5 or original_chain.overall_success_rate > 0.8:
            return None
        
        # Create evolved chain
        new_steps = []
        for step in original_chain.steps:
            if step.success_rate < 0.5:  # Poor performing step
                # Try to improve this step
                improved_content = self._improve_reasoning_step(step, problem_type)
                new_step = ReasoningStep(
                    step_type=step.step_type,
                    content=improved_content,
                    confidence=0.0,
                    success_rate=0.0,
                    usage_count=0
                )
                new_steps.append(new_step)
            else:
                # Keep good steps
                new_steps.append(step)
        
        # Add new reasoning steps if pattern is too simple
        if len(new_steps) < 4 and original_chain.overall_success_rate < 0.6:
            new_steps = self._add_reasoning_steps(new_steps, problem_type)
        
        # Create new chain
        new_chain_id = f"{problem_type}_{len(self.reasoning_chains):03d}"
        evolved_chain = ReasoningChain(
            chain_id=new_chain_id,
            problem_type=problem_type,
            steps=new_steps,
            overall_success_rate=0.0,
            improvement_count=0
        )
        
        # Record improvement
        self.improvement_history.append({
            "original_chain": chain_id,
            "evolved_chain": new_chain_id,
            "original_performance": original_chain.overall_success_rate,
            "problem_type": problem_type,
            "improvement_type": "reasoning_evolution"
        })
        
        self.reasoning_chains[new_chain_id] = evolved_chain
        return evolved_chain
    
    def _improve_reasoning_step(self, step: ReasoningStep, problem_type: str) -> str:
        """Improve a specific reasoning step based on effectiveness data"""
        # Get most effective patterns for this step type and problem type
        effectiveness = self.step_effectiveness.get(step.step_type, {})
        best_effectiveness = effectiveness.get(problem_type, 0.0)
        
        if step.step_type == "analysis" and best_effectiveness < 0.6:
            return "Systematically decompose the problem into core components and identify critical constraints"
        elif step.step_type == "hypothesis" and best_effectiveness < 0.6:
            return "Generate multiple solution candidates and rank them by feasibility and impact"
        elif step.step_type == "verification" and best_effectiveness < 0.6:
            return "Test each hypothesis against known principles and edge cases"
        elif step.step_type == "conclusion" and best_effectiveness < 0.6:
            return "Synthesize the most robust solution with clear implementation steps"
        else:
            # Default improvement: make more specific
            return f"Apply systematic {step.step_type} with focus on {problem_type}-specific factors"
    
    def _add_reasoning_steps(self, current_steps: List[ReasoningStep], problem_type: str) -> List[ReasoningStep]:
        """Add additional reasoning steps for complex problems"""
        enhanced_steps = current_steps.copy()
        
        # Add meta-reasoning step for complex problems
        if problem_type in ["technical", "diagnosis"]:
            meta_step = ReasoningStep(
                step_type="meta_analysis",
                content="Evaluate the reasoning process itself and identify potential blind spots",
                confidence=0.0
            )
            enhanced_steps.insert(-1, meta_step)  # Insert before conclusion
        
        return enhanced_steps
    
    def get_reasoning_evolution_metrics(self) -> Dict[str, any]:
        """Get metrics showing how reasoning has evolved"""
        total_chains = len(self.reasoning_chains)
        evolved_chains = len(self.improvement_history)
        
        avg_performance = sum(chain.overall_success_rate for chain in self.reasoning_chains.values()) / max(total_chains, 1)
        
        # Get step effectiveness trends
        step_effectiveness_avg = {}
        for step_type, problems in self.step_effectiveness.items():
            step_effectiveness_avg[step_type] = sum(problems.values()) / max(len(problems), 1)
        
        return {
            "total_reasoning_chains": total_chains,
            "evolved_chains": evolved_chains,
            "evolution_rate": evolved_chains / max(total_chains, 1),
            "average_chain_performance": avg_performance,
            "step_effectiveness": step_effectiveness_avg,
            "improvement_history": len(self.improvement_history),
            "reasoning_sophistication": avg_performance * (1 + evolved_chains * 0.1)
        }
    
    def save_reasoning_memory(self, filepath: str = "reasoning_memory.json"):
        """Save learned reasoning patterns"""
        memory_data = {
            "reasoning_chains": {
                chain_id: {
                    "chain_id": chain.chain_id,
                    "problem_type": chain.problem_type,
                    "steps": [
                        {
                            "step_type": step.step_type,
                            "content": step.content,
                            "success_rate": step.success_rate,
                            "usage_count": step.usage_count
                        } for step in chain.steps
                    ],
                    "overall_success_rate": chain.overall_success_rate,
                    "improvement_count": chain.improvement_count
                } for chain_id, chain in self.reasoning_chains.items()
            },
            "step_effectiveness": self.step_effectiveness,
            "improvement_history": self.improvement_history
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save reasoning memory: {e}")
    
    def load_reasoning_memory(self, filepath: str = "reasoning_memory.json"):
        """Load previously learned reasoning patterns"""
        try:
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            # Reconstruct reasoning chains
            for chain_id, chain_data in memory_data.get("reasoning_chains", {}).items():
                steps = [
                    ReasoningStep(
                        step_type=step_data["step_type"],
                        content=step_data["content"],
                        success_rate=step_data.get("success_rate", 0.0),
                        usage_count=step_data.get("usage_count", 0)
                    ) for step_data in chain_data["steps"]
                ]
                
                chain = ReasoningChain(
                    chain_id=chain_data["chain_id"],
                    problem_type=chain_data["problem_type"],
                    steps=steps,
                    overall_success_rate=chain_data.get("overall_success_rate", 0.0),
                    improvement_count=chain_data.get("improvement_count", 0)
                )
                
                self.reasoning_chains[chain_id] = chain
            
            # Load effectiveness data
            self.step_effectiveness = memory_data.get("step_effectiveness", {})
            self.improvement_history = memory_data.get("improvement_history", [])
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Create initial memory file
            self.save_reasoning_memory(filepath)


class ReasoningTaskEnvironment:
    """Environment for testing reasoning improvements"""
    
    def __init__(self):
        self.meta_engine = MetaReasoningEngine()
        self.meta_engine.load_reasoning_memory()
    
    def get_test_problems(self) -> List[Dict]:
        """Get test problems for different reasoning types"""
        return [
            {
                "id": "tech_oauth",
                "type": "technical", 
                "problem": "Design a secure OAuth2 implementation for a distributed microservices architecture",
                "success_criteria": ["security", "scalability", "implementation_clarity"]
            },
            {
                "id": "research_ai_safety",
                "type": "research",
                "problem": "Research the current state of AI alignment and safety measures in large language models",
                "success_criteria": ["comprehensiveness", "accuracy", "source_quality"]
            },
            {
                "id": "diag_performance",
                "type": "diagnosis", 
                "problem": "A web application suddenly has 5x slower response times. Diagnose the root cause",
                "success_criteria": ["systematic_approach", "root_cause_identification", "solution_feasibility"]
            },
            {
                "id": "tech_scaling",
                "type": "technical",
                "problem": "Design a system to handle 10M concurrent users with 99.99% uptime",
                "success_criteria": ["architecture_soundness", "scalability", "reliability_measures"]
            },
            {
                "id": "research_quantum",
                "type": "research", 
                "problem": "Analyze the potential impact of quantum computing on current cryptographic standards",
                "success_criteria": ["technical_depth", "timeline_accuracy", "practical_implications"]
            }
        ]
    
    def evaluate_reasoning_quality(self, reasoning_output: str, problem: Dict) -> Tuple[float, List[Tuple[int, bool]]]:
        """Evaluate quality of reasoning output"""
        # Simple evaluation (in production, would use LLM judge)
        score = 0.0
        step_results = []
        
        # Check for reasoning structure
        reasoning_steps = self._extract_reasoning_steps(reasoning_output)
        
        for i, step in enumerate(reasoning_steps):
            step_quality = self._evaluate_step_quality(step, problem)
            step_results.append((i, step_quality > 0.6))
            score += step_quality
        
        overall_score = score / max(len(reasoning_steps), 1)
        return overall_score, step_results
    
    def _extract_reasoning_steps(self, output: str) -> List[str]:
        """Extract individual reasoning steps from output"""
        # Look for numbered steps or clear reasoning structure
        steps = re.split(r'\d+\.|Step \d+:|•|→', output)
        return [step.strip() for step in steps if step.strip()]
    
    def _evaluate_step_quality(self, step: str, problem: Dict) -> float:
        """Evaluate quality of individual reasoning step"""
        # Simple heuristic evaluation
        quality = 0.0
        
        # Length suggests thoroughness
        if len(step) > 50:
            quality += 0.3
        
        # Technical terms suggest depth
        technical_terms = ["architecture", "system", "implementation", "analysis", "approach"]
        if any(term in step.lower() for term in technical_terms):
            quality += 0.4
        
        # Problem-specific relevance
        problem_keywords = problem["problem"].lower().split()
        relevance = sum(1 for word in problem_keywords if word in step.lower()) / len(problem_keywords)
        quality += relevance * 0.3
        
        return min(quality, 1.0)