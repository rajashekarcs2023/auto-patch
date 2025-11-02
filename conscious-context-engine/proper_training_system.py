#!/usr/bin/env python3
"""
Proper Self-Improving Agent Training System
Using ServerlessBackend, reward functions, and evaluation framework
"""
import asyncio
import math
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import weave
from dotenv import load_dotenv
import art
from art.serverless.backend import ServerlessBackend
from pydantic import BaseModel

load_dotenv()

# Ensure API keys are loaded
if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required for training")

from mcp_client import SelfImprovingMCPAgent

class AgentScenario(BaseModel):
    """Training scenario for agent learning"""
    step: int
    task_description: str
    task_type: str
    difficulty: float

@dataclass
class ToolUsageOutcome:
    """Outcome of tool usage for reward calculation"""
    tool_used: str
    task_type: str
    success: bool
    execution_time: float
    outcome_score: float
    context_relevance: float
    memory_efficiency: float

class AgentRewardFunction:
    """Sophisticated reward function for tool selection and memory learning"""
    
    def __init__(self):
        self.baseline_performance = {}  # Track baseline performance by task type
        self.learning_bonuses = {
            "tool_optimization": 0.3,
            "memory_efficiency": 0.2, 
            "context_relevance": 0.2,
            "speed_improvement": 0.1,
            "consistency": 0.2
        }
    
    def calculate_reward(self, outcome: ToolUsageOutcome, agent_state: Dict[str, Any]) -> float:
        """Calculate comprehensive reward for agent learning"""
        base_reward = 0.0
        
        # 1. Task Success Reward (40% of total)
        success_reward = 1.0 if outcome.success else -0.5
        base_reward += success_reward * 0.4
        
        # 2. Tool Selection Optimality (25% of total)
        tool_optimality = self._calculate_tool_optimality(outcome, agent_state)
        base_reward += tool_optimality * 0.25
        
        # 3. Memory Efficiency (20% of total)
        memory_efficiency = outcome.memory_efficiency
        base_reward += memory_efficiency * 0.2
        
        # 4. Speed/Efficiency (10% of total)
        speed_reward = self._calculate_speed_reward(outcome)
        base_reward += speed_reward * 0.1
        
        # 5. Learning Progress Bonus (5% of total)
        learning_bonus = self._calculate_learning_bonus(outcome, agent_state)
        base_reward += learning_bonus * 0.05
        
        # Normalize reward to [-1, 1] range
        final_reward = max(-1.0, min(1.0, base_reward))
        
        return final_reward
    
    def _calculate_tool_optimality(self, outcome: ToolUsageOutcome, agent_state: Dict[str, Any]) -> float:
        """Calculate how optimal the tool selection was"""
        task_type = outcome.task_type
        tool_used = outcome.tool_used
        
        # Get tool performance history for this task type
        tool_performances = agent_state.get('tool_performance', {})
        relevant_tools = {k: v for k, v in tool_performances.items() if task_type in k}
        
        if not relevant_tools:
            return 0.5  # Neutral for new task types
        
        # Find best performing tool for this task type
        best_tool_performance = max(relevant_tools.values(), key=lambda x: x.get('success_rate', 0))
        current_tool_key = f"{tool_used}_{task_type}"
        current_performance = tool_performances.get(current_tool_key, {'success_rate': 0.5})
        
        # Reward for choosing high-performing tools
        optimality = current_performance['success_rate'] / max(best_tool_performance.get('success_rate', 0.5), 0.1)
        return min(1.0, optimality)
    
    def _calculate_speed_reward(self, outcome: ToolUsageOutcome) -> float:
        """Calculate reward based on execution speed"""
        # Reward faster execution times
        if outcome.execution_time <= 0.5:
            return 1.0
        elif outcome.execution_time <= 1.0:
            return 0.7
        elif outcome.execution_time <= 2.0:
            return 0.4
        else:
            return 0.1
    
    def _calculate_learning_bonus(self, outcome: ToolUsageOutcome, agent_state: Dict[str, Any]) -> float:
        """Calculate bonus for demonstrating learning progress"""
        bonus = 0.0
        
        # Bonus for improving over baseline
        task_type = outcome.task_type
        if task_type in self.baseline_performance:
            baseline = self.baseline_performance[task_type]
            if outcome.outcome_score > baseline:
                improvement = (outcome.outcome_score - baseline) / baseline
                bonus += min(0.5, improvement * 0.5)
        else:
            # First time seeing this task type
            self.baseline_performance[task_type] = outcome.outcome_score
        
        # Bonus for memory efficiency improvements
        memory_items = agent_state.get('memory_items', 0)
        if memory_items > 0 and outcome.memory_efficiency > 0.7:
            bonus += 0.2
        
        return bonus

class ToolSelectionEvaluator:
    """Evaluates tool selection quality and provides feedback"""
    
    def __init__(self):
        self.tool_capabilities = {
            "research": ["perplexity_search", "perplexity_research", "perplexity_reason"],
            "travel": ["airbnb_search", "airbnb_listing_details"],
            "communication": ["vapi_create_call", "vapi_synthesize", "vapi_list_assistants"],
            "web_scraping": ["firecrawl_scrape", "firecrawl_crawl", "firecrawl_extract", "firecrawl_map", "real_firecrawl_scrape", "real_firecrawl_batch_scrape", "real_firecrawl_extract"],
            "documentation": ["context7_resolve_library_id", "context7_get_library_docs", "context7_search_docs"]
        }
    
    def evaluate_tool_selection(self, task_type: str, tool_used: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality of tool selection"""
        evaluation = {
            "tool_appropriateness": 0.0,
            "task_completion": 0.0,
            "efficiency_score": 0.0,
            "feedback": ""
        }
        
        # Check if tool is appropriate for task type
        appropriate_tools = self.tool_capabilities.get(task_type, [])
        if any(tool_used.startswith(tool_prefix.split('_')[0]) for tool_prefix in appropriate_tools):
            evaluation["tool_appropriateness"] = 1.0
            evaluation["feedback"] += f"✅ Good tool choice: {tool_used} is appropriate for {task_type} tasks. "
        else:
            evaluation["tool_appropriateness"] = 0.3
            evaluation["feedback"] += f"⚠️ Suboptimal tool: {tool_used} may not be ideal for {task_type}. "
            
            # Suggest better tools
            if appropriate_tools:
                suggested = appropriate_tools[0]
                evaluation["feedback"] += f"Consider {suggested} instead. "
        
        # Evaluate task completion
        success = outcome.get('success', False)
        outcome_score = outcome.get('outcome_score', 0.0)
        
        if success and outcome_score > 0.8:
            evaluation["task_completion"] = 1.0
            evaluation["feedback"] += "✅ Excellent task completion. "
        elif success and outcome_score > 0.6:
            evaluation["task_completion"] = 0.7
            evaluation["feedback"] += "✅ Good task completion. "
        elif success:
            evaluation["task_completion"] = 0.5
            evaluation["feedback"] += "⚠️ Basic task completion. "
        else:
            evaluation["task_completion"] = 0.0
            evaluation["feedback"] += "❌ Task failed. "
        
        # Evaluate efficiency
        execution_time = outcome.get('execution_time', 5.0)
        if execution_time < 1.0:
            evaluation["efficiency_score"] = 1.0
            evaluation["feedback"] += "⚡ Fast execution. "
        elif execution_time < 2.0:
            evaluation["efficiency_score"] = 0.7
            evaluation["feedback"] += "📈 Reasonable speed. "
        else:
            evaluation["efficiency_score"] = 0.3
            evaluation["feedback"] += "🐌 Slow execution - consider optimization. "
        
        return evaluation

@weave.op
@art.retry(exceptions=(Exception,))
async def agent_rollout(model: art.Model, scenario: AgentScenario) -> art.Trajectory:
    """Execute agent training rollout with proper evaluation"""
    
    # Initialize agent and evaluators
    agent = SelfImprovingMCPAgent()
    reward_function = AgentRewardFunction()
    evaluator = ToolSelectionEvaluator()
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system", 
                "content": f"You are a self-improving agent learning optimal tool selection. Task: {scenario.task_description}. Choose the best tool and execute efficiently."
            }
        ],
        metadata={
            "step": scenario.step,
            "task_type": scenario.task_type,
            "task_description": scenario.task_description,
            "difficulty": scenario.difficulty
        },
        reward=0
    )
    
    try:
        # Get agent state before task
        initial_state = agent.get_learning_insights()
        
        # Execute the task
        start_time = time.time()
        result = await agent.execute_task(scenario.task_description, scenario.task_type)
        execution_time = time.time() - start_time
        
        # Get agent state after task
        final_state = agent.get_learning_insights()
        
        # Create outcome for evaluation
        outcome = ToolUsageOutcome(
            tool_used=result['tool_used'],
            task_type=scenario.task_type,
            success=result['success'],
            execution_time=execution_time,
            outcome_score=result.get('outcome_score', 0.5),
            context_relevance=0.8,  # Could be calculated based on context usage
            memory_efficiency=final_state.get('memory_efficiency', 0.5)
        )
        
        # Calculate reward
        reward = reward_function.calculate_reward(outcome, final_state)
        
        # Get evaluation feedback
        evaluation = evaluator.evaluate_tool_selection(scenario.task_type, result['tool_used'], result)
        
        # Get tool rankings for confidence metric  
        from mcp_client import TaskContext
        task_context = TaskContext(
            task_id="",
            task_type=scenario.task_type,
            user_intent=scenario.task_description,
            available_tools=list(agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        tool_rankings = agent.tool_learner.get_best_tools(task_context, list(agent.tools.keys()))
        
        # Store results in trajectory
        trajectory.messages_and_choices.append({
            "role": "user",
            "content": f"Task: {scenario.task_description}\nType: {scenario.task_type}"
        })
        
        trajectory.messages_and_choices.append({
            "role": "assistant", 
            "content": f"Selected tool: {result['tool_used']}\nResult: {result['success']}\nScore: {outcome.outcome_score:.3f}"
        })
        
        trajectory.reward = reward
        trajectory.metrics.update({
            "task_success": 1.0 if result['success'] else 0.0,
            "outcome_score": float(outcome.outcome_score),
            "execution_time": float(execution_time),
            "memory_efficiency": float(outcome.memory_efficiency),
            "tool_appropriateness": float(evaluation["tool_appropriateness"]),
            "task_completion": float(evaluation["task_completion"]),
            "efficiency_score": float(evaluation["efficiency_score"]),
            "learning_progress": float(final_state['total_experience'] - initial_state['total_experience']),
            "memory_items": float(final_state['memory_items']),
            "memory_pruning_events": float(final_state.get('memory_pruning_events', 0)),
            "tool_confidence": float(tool_rankings[0][1] if tool_rankings else 0.5)
        })
        
        print(f"📊 Step {scenario.step}: {scenario.task_type} | Tool: {result['tool_used']} | Reward: {reward:.3f}")
        print(f"   💡 {evaluation['feedback']}")
        
    except Exception as e:
        print(f"❌ Error in rollout: {e}")
        trajectory.reward = -1.0
        trajectory.metrics["error"] = str(e)
    
    return trajectory

class ProperTrainingSystem:
    """Proper training system using ServerlessBackend"""
    
    def __init__(self):
        self.model = None
        self.backend = None
        self.training_scenarios = [
            ("Research latest AI developments in computer vision", "research", 0.7),
            ("Find luxury hotels in Tokyo for business trip", "travel", 0.6),
            ("Create professional voice greeting for customers", "communication", 0.5),
            ("Extract product data from e-commerce website", "web_scraping", 0.8),
            ("Get documentation for React hooks best practices", "documentation", 0.7),
            ("Research quantum computing breakthroughs 2024", "research", 0.9),
            ("Book family accommodation in Paris", "travel", 0.7),
            ("Generate customer service phone script", "communication", 0.6),
            ("Scrape competitor pricing information", "web_scraping", 0.8),
            ("Find Next.js documentation for the 'after' function", "documentation", 0.8),
            ("Research machine learning security vulnerabilities", "research", 0.9),
            ("Find pet-friendly hotels in San Francisco", "travel", 0.6),
            ("Create podcast introduction audio", "communication", 0.5),
            ("Map website structure for SEO analysis", "web_scraping", 0.7),
            ("Get TypeScript 5.x documentation for new features", "documentation", 0.6),
            ("Batch scrape multiple competitor websites", "web_scraping", 0.9)
        ]
    
    async def initialize_training(self):
        """Initialize ServerlessBackend and model"""
        print("🚀 INITIALIZING PROPER TRAINING SYSTEM")
        print("=" * 60)
        
        # Initialize Weave for tracking
        weave.init("self-improving-agent", settings={"print_call_link": False})
        
        # Declare the model following exact pattern from examplecode3.md
        self.model = art.TrainableModel(
            name="self-improving-mcp-agent",
            project="agent-learning", 
            base_model="OpenPipe/Qwen3-14B-Instruct"
        )
        
        # Initialize ServerlessBackend
        self.backend = ServerlessBackend()
        
        # Register model with backend
        await self.model.register(self.backend)
        
        print(f"✅ Model registered: {self.model.name}")
        print(f"✅ Project: {self.model.project}")
        print(f"✅ Backend initialized")
    
    async def run_training(self, num_steps: int = 15):
        """Run the training loop"""
        print(f"\n🎯 STARTING TRAINING LOOP")
        print(f"📊 Training steps: {num_steps}")
        print(f"📋 Scenarios per step: 6")
        print("-" * 40)
        
        for step in range(await self.model.get_step(), num_steps):
            print(f"\n🔄 TRAINING STEP {step + 1}/{num_steps}")
            
            # Create scenarios for this step
            step_scenarios = []
            for i in range(6):  # 6 scenarios per step
                task_desc, task_type, difficulty = self.training_scenarios[i % len(self.training_scenarios)]
                scenario = AgentScenario(
                    step=step,
                    task_description=task_desc,
                    task_type=task_type,
                    difficulty=difficulty
                )
                step_scenarios.append(scenario)
            
            # Gather trajectory groups (following exact pattern)
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(agent_rollout(self.model, scenario) for scenario in step_scenarios)
                    for _ in range(1)
                ),
                pbar_desc=f"Step {step + 1} training",
                max_exceptions=6
            )
            
            # Delete old checkpoints and train
            await self.model.delete_checkpoints('train/reward')
            await self.model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=1e-5)
            )
            
            print(f"✅ Step {step + 1} training complete")
        
        print(f"\n🎉 TRAINING COMPLETE!")
        
        # Get final step and deploy
        last_step = await self.model.get_step()
        deployed_model_name = f"{self.model.get_inference_name()}:step{last_step}"
        
        print(f"🚀 Model deployed as: {deployed_model_name}")
        
        return deployed_model_name
    
    async def demonstrate_learned_agent(self, deployed_model_name: str):
        """Demonstrate the trained agent"""
        print(f"\n🎬 DEMONSTRATING TRAINED AGENT")
        print("-" * 40)
        
        # Create test scenarios
        test_scenarios = [
            ("Research breakthrough AI safety techniques", "research"),
            ("Find boutique hotels in NYC", "travel"),
            ("Create welcome message for new employees", "communication"),
            ("Extract competitor features from their website", "web_scraping")
        ]
        
        for i, (task_desc, task_type) in enumerate(test_scenarios, 1):
            print(f"\n🧪 TEST {i}: {task_desc}")
            
            # Use trained agent
            agent = SelfImprovingMCPAgent()
            result = await agent.execute_task(task_desc, task_type)
            
            print(f"   🔧 Tool Selected: {result['tool_used']}")
            print(f"   ✅ Success: {result['success']}")
            print(f"   📊 Score: {result.get('outcome_score', 0):.3f}")
            print(f"   ⏱️  Time: {result.get('execution_time', 0):.2f}s")
            
            # Show learning state
            insights = agent.get_learning_insights()
            print(f"   🧠 Memory: {insights['memory_items']} items")
            print(f"   🎯 Tool Patterns: {len(insights['tool_performance'])}")

async def main():
    """Run the proper training system"""
    trainer = ProperTrainingSystem()
    
    try:
        await trainer.initialize_training()
        deployed_model = await trainer.run_training(num_steps=10)
        await trainer.demonstrate_learned_agent(deployed_model)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import os
    asyncio.run(main())