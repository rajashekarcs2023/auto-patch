#!/usr/bin/env python3
"""
Debug Training System - Find out why rollouts are failing
Add extensive logging to see what's going wrong
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
import traceback

import art
from art.serverless.backend import ServerlessBackend

load_dotenv()

# Ensure API keys are loaded
if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required for training")

class AgentScenario(BaseModel):
    step: int
    task_description: str
    task_type: str
    optimal_tools: List[str]
    difficulty: float

AVAILABLE_TOOLS = [
    "firecrawl_scrape", "firecrawl_crawl", "firecrawl_extract", "firecrawl_map", 
    "perplexity_search", "perplexity_research", "airbnb_search", "vapi_synthesize"
]

AGENT_SCENARIOS = [
    {
        "task": "Research the latest AI safety developments",
        "task_type": "research",
        "optimal_tools": ["perplexity_search", "perplexity_research"],
        "difficulty": 0.8
    },
    {
        "task": "Find hotels in Tokyo",
        "task_type": "travel", 
        "optimal_tools": ["airbnb_search"],
        "difficulty": 0.6
    },
    {
        "task": "Create voice message",
        "task_type": "communication",
        "optimal_tools": ["vapi_synthesize"],
        "difficulty": 0.5
    },
    {
        "task": "Extract website data",
        "task_type": "web_scraping",
        "optimal_tools": ["firecrawl_scrape", "firecrawl_extract"],
        "difficulty": 0.7
    }
]

def calculate_dynamic_reward(chosen_tool: str, optimal_tools: List[str], task_type: str, difficulty: float) -> float:
    """Calculate dynamic reward with variation"""
    
    # Base reward based on tool appropriateness
    if chosen_tool in optimal_tools:
        base_reward = 0.9 + random.uniform(-0.1, 0.1)  # 0.8-1.0 for optimal
    elif any(opt.split('_')[0] in chosen_tool for opt in optimal_tools):
        base_reward = 0.6 + random.uniform(-0.2, 0.2)  # 0.4-0.8 for category match
    else:
        base_reward = 0.2 + random.uniform(-0.1, 0.3)  # 0.1-0.5 for poor choice
    
    # Difficulty adjustment
    difficulty_penalty = difficulty * 0.1
    
    # Add realistic variation
    final_reward = base_reward - difficulty_penalty + random.uniform(-0.05, 0.05)
    
    return max(0.0, min(1.0, final_reward))

@weave.op  
@art.retry(exceptions=(requests.ReadTimeout,))
async def debug_rollout(model: art.Model, scenario: AgentScenario) -> art.Trajectory:
    """Debug version with extensive logging"""
    
    print(f"\n🔍 DEBUG ROLLOUT - Step {scenario.step}")
    print(f"Task: {scenario.task_description}")
    print(f"Type: {scenario.task_type}")
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"""You are a self-improving MCP agent that learns optimal tool selection.

Available tools: {', '.join(AVAILABLE_TOOLS)}

Your task: {scenario.task_description}
Task type: {scenario.task_type}

Choose the most appropriate tool and respond with:
<tool_choice>[tool_name]</tool_choice>"""
            }
        ],
        metadata={
            "task_type": scenario.task_type,
            "step": scenario.step
        },
        reward=0
    )
    
    # Add user message
    trajectory.messages_and_choices.append({
        "role": "user", 
        "content": f"Task: {scenario.task_description}\nChoose the best tool."
    })
    
    try:
        print(f"🌐 Model inference URL: {model.inference_base_url}")
        print(f"🔑 API key exists: {bool(model.inference_api_key)}")
        print(f"📝 Model name: {model.get_inference_name()}")
        
        # Create OpenAI client
        client = AsyncOpenAI(
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
        )
        
        print("🔄 Making LLM call...")
        
        # Get messages
        messages = trajectory.messages()
        print(f"📄 Messages count: {len(messages)}")
        
        # Make LLM call with timeout
        chat_completion = await asyncio.wait_for(
            client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
                temperature=0.7
            ),
            timeout=30.0
        )
        
        print("✅ LLM call successful!")
        
        choice = chat_completion.choices[0]
        content = choice.message.content
        print(f"🤖 Model response: {content}")
        
        # Add response to trajectory
        trajectory.messages_and_choices.append(choice)
        
        # Parse tool choice
        chosen_tool = "firecrawl_scrape"  # default
        if "<tool_choice>" in content and "</tool_choice>" in content:
            start = content.find("<tool_choice>") + len("<tool_choice>")
            end = content.find("</tool_choice>")
            chosen_tool = content[start:end].strip()
        
        print(f"🔧 Chosen tool: {chosen_tool}")
        
        # Calculate dynamic reward
        reward = calculate_dynamic_reward(chosen_tool, scenario.optimal_tools, scenario.task_type, scenario.difficulty)
        trajectory.reward = reward
        
        print(f"🏆 Reward: {reward:.3f}")
        
        # Add metrics
        trajectory.metrics.update({
            "chosen_tool": chosen_tool,
            "task_difficulty": scenario.difficulty,
            "optimal_choice": chosen_tool in scenario.optimal_tools,
            "response_length": len(content) if content else 0
        })
        
        print(f"✅ Rollout successful - Reward: {reward:.3f}")
        return trajectory
        
    except asyncio.TimeoutError:
        print("❌ ERROR: LLM call timed out after 30 seconds")
        trajectory.reward = -1.0
        trajectory.metrics["error"] = "timeout"
        return trajectory
        
    except Exception as e:
        print(f"❌ ERROR in rollout: {type(e).__name__}: {e}")
        print(f"📋 Full traceback:")
        traceback.print_exc()
        
        trajectory.reward = -1.0
        trajectory.metrics["error"] = str(e)
        return trajectory

async def test_single_rollout():
    """Test a single rollout to debug issues"""
    print("🧪 TESTING SINGLE ROLLOUT")
    print("=" * 50)
    
    # Initialize model
    weave.init("debug-agent", settings={"print_call_link": False})
    
    model = art.TrainableModel(
        name="debug-mcp-agent",
        project="debug-learning",
        base_model="OpenPipe/Qwen3-14B-Instruct"
    )
    
    backend = ServerlessBackend()
    await model.register(backend)
    
    print(f"✅ Model registered: {model.name}")
    print(f"✅ Inference URL: {model.inference_base_url}")
    print(f"✅ Model inference name: {model.get_inference_name()}")
    
    # Create test scenario
    scenario = AgentScenario(
        step=0,
        task_description="Research AI safety developments",
        task_type="research",
        optimal_tools=["perplexity_search", "perplexity_research"],
        difficulty=0.8
    )
    
    # Run single rollout with debug
    result = await debug_rollout(model, scenario)
    
    print(f"\n📊 FINAL RESULT:")
    print(f"Reward: {result.reward}")
    print(f"Metrics: {result.metrics}")
    print(f"Messages count: {len(result.messages_and_choices)}")

async def test_training_step():
    """Test a single training step with multiple rollouts"""
    print("\n🏋️ TESTING TRAINING STEP")
    print("=" * 50)
    
    # Initialize model  
    weave.init("debug-agent", settings={"print_call_link": False})
    
    model = art.TrainableModel(
        name="debug-mcp-agent", 
        project="debug-learning",
        base_model="OpenPipe/Qwen3-14B-Instruct"
    )
    
    backend = ServerlessBackend()
    await model.register(backend)
    
    # Create scenarios
    scenarios = []
    for i in range(6):  # Test with 6 instead of 18
        scenario_data = random.choice(AGENT_SCENARIOS)
        scenario = AgentScenario(
            step=0,
            task_description=scenario_data["task"],
            task_type=scenario_data["task_type"], 
            optimal_tools=scenario_data["optimal_tools"],
            difficulty=scenario_data["difficulty"]
        )
        scenarios.append(scenario)
    
    print(f"🎯 Testing {len(scenarios)} rollouts...")
    
    # Run training step
    try:
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(debug_rollout(model, scenario) for scenario in scenarios)
                for _ in range(1)
            ),
            pbar_desc="Debug training",
            max_exceptions=6
        )
        
        print(f"✅ Trajectory groups gathered: {len(train_groups)}")
        
        # Check rewards
        for group in train_groups:
            for trajectory in group.trajectories:
                print(f"Trajectory reward: {trajectory.reward}")
                print(f"Trajectory metrics: {trajectory.metrics}")
        
        # Try training
        await model.delete_checkpoints('train/reward')
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5)
        )
        
        print("✅ Training step completed successfully!")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUGGING TRAINING ISSUES")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_single_rollout())
    asyncio.run(test_training_step())