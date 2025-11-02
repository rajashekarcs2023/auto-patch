#!/usr/bin/env python3
"""
Working Training System - Fixed metrics to only use numeric values
Real LLM calls with dynamic rewards that should show variation in W&B
"""
import asyncio
import json
import math
import random
import requests
import weave
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

import art
from art.serverless.backend import ServerlessBackend

load_dotenv()

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required for training")

class AgentScenario(BaseModel):
    step: int
    task_description: str
    task_type: str
    optimal_tools: List[str]
    difficulty: float

# Tool categories for scoring
TOOL_CATEGORIES = {
    "research": ["perplexity"],
    "travel": ["airbnb"], 
    "communication": ["vapi"],
    "web_scraping": ["firecrawl"],
    "documentation": ["context7"]
}

AVAILABLE_TOOLS = [
    "firecrawl_scrape", "firecrawl_crawl", "firecrawl_extract", "firecrawl_map", 
    "perplexity_search", "perplexity_research", "perplexity_reason",
    "airbnb_search", "airbnb_listing_details",
    "vapi_synthesize", "vapi_create_call", "vapi_create_assistant",
    "context7_resolve_library_id", "context7_get_library_docs"
]

AGENT_SCENARIOS = [
    {
        "task": "Research the latest AI safety developments in reinforcement learning",
        "task_type": "research",
        "optimal_tools": ["perplexity_search", "perplexity_research"],
        "difficulty": 0.8
    },
    {
        "task": "Find luxury accommodation in Tokyo for a 5-day business trip",
        "task_type": "travel", 
        "optimal_tools": ["airbnb_search", "airbnb_listing_details"],
        "difficulty": 0.6
    },
    {
        "task": "Create a professional voice greeting message for customer service",
        "task_type": "communication",
        "optimal_tools": ["vapi_synthesize", "vapi_create_call"],
        "difficulty": 0.5
    },
    {
        "task": "Extract product pricing and features from competitor's e-commerce site",
        "task_type": "web_scraping",
        "optimal_tools": ["firecrawl_scrape", "firecrawl_extract"],
        "difficulty": 0.9
    },
    {
        "task": "Get comprehensive documentation for React 18 hooks with examples",
        "task_type": "documentation", 
        "optimal_tools": ["context7_resolve_library_id", "context7_get_library_docs"],
        "difficulty": 0.7
    }
]

def calculate_tool_score(chosen_tool: str, optimal_tools: List[str], task_type: str) -> float:
    """Calculate tool appropriateness score (0-1)"""
    if chosen_tool in optimal_tools:
        return 1.0
    
    # Check category match
    expected_categories = TOOL_CATEGORIES.get(task_type, [])
    if any(cat in chosen_tool for cat in expected_categories):
        return 0.7
    
    return 0.2

def calculate_dynamic_reward(chosen_tool: str, optimal_tools: List[str], task_type: str, difficulty: float) -> tuple:
    """Calculate dynamic reward with realistic variation"""
    
    # Tool appropriateness (0-1)
    tool_score = calculate_tool_score(chosen_tool, optimal_tools, task_type)
    
    # Task completion simulation (0.5-1.0 with variation)
    base_completion = 0.7 + random.uniform(-0.1, 0.2)
    difficulty_penalty = difficulty * 0.1
    completion_score = max(0.5, min(1.0, base_completion - difficulty_penalty))
    
    # Efficiency score (0.4-1.0 based on tool choice)
    if tool_score > 0.8:
        efficiency = 0.8 + random.uniform(0, 0.2)
    elif tool_score > 0.5:
        efficiency = 0.6 + random.uniform(0, 0.2)
    else:
        efficiency = 0.4 + random.uniform(0, 0.3)
    
    # Composite reward (like 2048 example)
    final_reward = (tool_score * 0.5) + (completion_score * 0.3) + (efficiency * 0.2)
    
    # Add small random variation for realistic learning curves
    final_reward += random.uniform(-0.05, 0.05)
    final_reward = max(0.0, min(1.0, final_reward))
    
    return final_reward, tool_score, completion_score, efficiency

def tool_name_to_id(tool_name: str) -> int:
    """Convert tool name to numeric ID for metrics"""
    try:
        return AVAILABLE_TOOLS.index(tool_name)
    except ValueError:
        return 0  # Default to first tool

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def working_rollout(model: art.Model, scenario: AgentScenario) -> art.Trajectory:
    """Working rollout with proper numeric metrics only"""
    
    # Create OpenAI client
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"""You are a self-improving MCP agent that learns optimal tool selection.

Available tools: {', '.join(AVAILABLE_TOOLS)}

Your task: {scenario.task_description}
Task type: {scenario.task_type}

Analyze the task and choose the most appropriate tool. Consider tool capabilities and task requirements.

Respond with your tool choice in this format:
<tool_choice>[tool_name]</tool_choice>"""
            }
        ],
        metadata={
            "task_type": scenario.task_type,
            "step": scenario.step,
            "difficulty": scenario.difficulty
        },
        reward=0
    )
    
    # Add user message
    trajectory.messages_and_choices.append({
        "role": "user", 
        "content": f"Task: {scenario.task_description}\nType: {scenario.task_type}\nChoose the best tool."
    })
    
    try:
        # Make LLM call
        messages = trajectory.messages()
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=128,
            messages=messages,
            model=model.get_inference_name(),
            temperature=0.7
        )
        
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        
        # Add response to trajectory
        trajectory.messages_and_choices.append(choice)
        
        # Parse tool choice
        chosen_tool = "firecrawl_scrape"  # default
        if "<tool_choice>" in content and "</tool_choice>" in content:
            start = content.find("<tool_choice>") + len("<tool_choice>")
            end = content.find("</tool_choice>")
            parsed_tool = content[start:end].strip()
            if parsed_tool in AVAILABLE_TOOLS:
                chosen_tool = parsed_tool
        
        # Calculate dynamic rewards and scores
        reward, tool_score, completion_score, efficiency = calculate_dynamic_reward(
            chosen_tool, scenario.optimal_tools, scenario.task_type, scenario.difficulty
        )
        
        trajectory.reward = reward
        
        # Only use NUMERIC metrics (no strings!)
        trajectory.metrics.update({
            "tool_id": float(tool_name_to_id(chosen_tool)),  # Convert tool to numeric ID
            "tool_score": float(tool_score),
            "completion_score": float(completion_score), 
            "efficiency_score": float(efficiency),
            "task_difficulty": float(scenario.difficulty),
            "optimal_choice": float(1.0 if chosen_tool in scenario.optimal_tools else 0.0),
            "response_length": float(len(content)),
            "step": float(scenario.step)
        })
        
        print(f"📊 Step {scenario.step}: {scenario.task_type} | Tool: {chosen_tool} | Reward: {reward:.3f} | Optimal: {chosen_tool in scenario.optimal_tools}")
        
        return trajectory
        
    except Exception as e:
        print(f"❌ Error in rollout: {e}")
        trajectory.reward = -1.0
        trajectory.metrics.update({
            "error": 1.0,
            "step": float(scenario.step)
        })
        return trajectory

class WorkingTrainingSystem:
    """Working training system with proper numeric metrics"""
    
    def __init__(self):
        self.model = None
        self.backend = None
        
    async def initialize_training(self):
        """Initialize model and backend"""
        print("🚀 INITIALIZING WORKING TRAINING SYSTEM")
        print("Using only numeric metrics to avoid 422 errors")
        print("=" * 60)
        
        weave.init("working-agent", settings={"print_call_link": False})
        
        self.model = art.TrainableModel(
            name="working-mcp-agent",
            project="working-learning",
            base_model="OpenPipe/Qwen3-14B-Instruct"
        )
        
        self.backend = ServerlessBackend()
        await self.model.register(self.backend)
        
        print(f"✅ Model registered: {self.model.name}")
        print(f"✅ Base model: OpenPipe/Qwen3-14B-Instruct")
        print(f"✅ Inference URL: {self.model.inference_base_url}")
    
    async def run_training(self, num_steps: int = 20):
        """Run real training with working metrics"""
        print(f"\n🎯 STARTING WORKING TRAINING")
        print(f"📊 Training steps: {num_steps}")
        print(f"📋 Rollouts per step: 12 (reduced for stability)")
        print("-" * 60)
        
        for i in range(await self.model.get_step(), num_steps):
            print(f"\n🔄 TRAINING STEP {i + 1}/{num_steps}")
            
            # Create diverse scenarios
            step_scenarios = []
            for j in range(12):  # 12 rollouts for faster iteration
                scenario_data = random.choice(AGENT_SCENARIOS)
                scenario = AgentScenario(
                    step=i,
                    task_description=scenario_data["task"],
                    task_type=scenario_data["task_type"],
                    optimal_tools=scenario_data["optimal_tools"],
                    difficulty=scenario_data["difficulty"]
                )
                step_scenarios.append(scenario)
            
            # Run training step
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(working_rollout(self.model, scenario) for scenario in step_scenarios)
                    for _ in range(1)
                ),
                pbar_desc=f"Step {i + 1} training",
                max_exceptions=12
            )
            
            # Train model
            await self.model.delete_checkpoints('train/reward')
            await self.model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=1e-5)
            )
            
            # Show step summary
            total_reward = sum(
                traj.reward for group in train_groups for traj in group.trajectories
            )
            avg_reward = total_reward / (len(train_groups) * len(step_scenarios))
            print(f"✅ Step {i + 1} complete - Avg Reward: {avg_reward:.3f}")
        
        print(f"\n🎉 TRAINING COMPLETE!")
        last_step = await self.model.get_step()
        deployed_model = f"{self.model.get_inference_name()}:step{last_step}"
        print(f"🚀 Model deployed: {deployed_model}")
        
        return deployed_model
    
    async def test_trained_agent(self, deployed_model: str):
        """Test the trained agent"""
        print(f"\n🎬 TESTING TRAINED AGENT")
        print(f"Model: {deployed_model}")
        print("-" * 60)
        
        client = AsyncOpenAI(
            base_url=self.model.inference_base_url,
            api_key=self.model.inference_api_key,
        )
        
        test_cases = [
            ("Research breakthrough AI safety techniques", "research"),
            ("Find boutique hotels in NYC", "travel"),
            ("Create welcome voice message", "communication"),
            ("Extract competitor pricing data", "web_scraping"),
            ("Get Next.js 14 documentation", "documentation")
        ]
        
        for i, (task, task_type) in enumerate(test_cases, 1):
            print(f"\n🧪 TEST {i}: {task}")
            
            messages = [
                {
                    "role": "system",
                    "content": f"You are a trained MCP agent. Available tools: {', '.join(AVAILABLE_TOOLS)}\n\nChoose the best tool: <tool_choice>[tool_name]</tool_choice>"
                },
                {
                    "role": "user",
                    "content": f"Task: {task}\nType: {task_type}"
                }
            ]
            
            try:
                response = await client.chat.completions.create(
                    model=deployed_model,
                    messages=messages,
                    max_completion_tokens=128
                )
                
                content = response.choices[0].message.content
                
                # Parse tool
                tool = "unknown"
                if "<tool_choice>" in content and "</tool_choice>" in content:
                    start = content.find("<tool_choice>") + len("<tool_choice>")
                    end = content.find("</tool_choice>")
                    tool = content[start:end].strip()
                
                print(f"   🔧 Tool: {tool}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

async def main():
    """Run working training system"""
    trainer = WorkingTrainingSystem()
    
    try:
        await trainer.initialize_training()
        deployed_model = await trainer.run_training(num_steps=20)
        await trainer.test_trained_agent(deployed_model)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())