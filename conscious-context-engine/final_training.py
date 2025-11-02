#!/usr/bin/env python3
"""
Final Training System - Following examplecode3.md EXACTLY
Multi-turn agent like 2048 game with multiple tool choices per trajectory
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

random.seed(42)  # Like in example

class AgentScenario(BaseModel):
    step: int

AVAILABLE_TOOLS = [
    "firecrawl_scrape", "firecrawl_extract", "perplexity_search", 
    "perplexity_research", "airbnb_search", "vapi_synthesize",
    "context7_get_library_docs"
]

# Task scenarios that build on each other (like 2048 moves)
TASK_SCENARIOS = [
    {
        "task": "Research AI safety developments",
        "type": "research",
        "optimal": ["perplexity_search", "perplexity_research"],
        "next_tasks": [
            {"task": "Extract specific findings from research papers", "type": "web_scraping", "optimal": ["firecrawl_extract"]},
            {"task": "Get documentation on AI safety frameworks", "type": "documentation", "optimal": ["context7_get_library_docs"]}
        ]
    },
    {
        "task": "Find hotels in Tokyo",
        "type": "travel", 
        "optimal": ["airbnb_search"],
        "next_tasks": [
            {"task": "Create booking confirmation voice message", "type": "communication", "optimal": ["vapi_synthesize"]},
            {"task": "Research Tokyo travel tips", "type": "research", "optimal": ["perplexity_search"]}
        ]
    },
    {
        "task": "Extract competitor pricing data",
        "type": "web_scraping",
        "optimal": ["firecrawl_scrape", "firecrawl_extract"],
        "next_tasks": [
            {"task": "Research market analysis techniques", "type": "research", "optimal": ["perplexity_research"]},
            {"task": "Create pricing analysis voice report", "type": "communication", "optimal": ["vapi_synthesize"]}
        ]
    }
]

def calculate_move_score(chosen_tool: str, optimal_tools: List[str], move_number: int) -> float:
    """Calculate score for this move (like 2048 scoring)"""
    base_score = 2 if chosen_tool in optimal_tools else 0
    
    # Bonus for consecutive good moves
    move_bonus = move_number * 0.1 if chosen_tool in optimal_tools else 0
    
    # Random variation
    variation = random.uniform(-0.1, 0.1)
    
    return max(0, base_score + move_bonus + variation)

def calculate_final_reward(total_score: float, moves_made: int) -> float:
    """Calculate final reward like 2048 example"""
    if moves_made == 0:
        return -1.0
    
    # Scale like in 2048 example (logarithmic)
    if total_score >= 10:  # "Winning" threshold
        return 2.0  # Double reward like 2048 win
    else:
        # Scale between 0 and 1 based on performance
        max_score_reward = math.log(max(total_score, 1), 2) / math.log(10, 2)
        return max_score_reward * 0.8  # Scale to 0-0.8 range

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, scenario: AgentScenario) -> art.Trajectory:
    """Multi-turn rollout like 2048 example"""
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Pick random starting scenario
    current_scenario = random.choice(TASK_SCENARIOS)
    move_number = 0
    total_score = 0
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"""You are an excellent MCP agent that learns optimal tool selection. Available tools: {', '.join(AVAILABLE_TOOLS)}

For each task, choose the most appropriate tool and respond with: <tool_choice>tool_name</tool_choice>"""
            }
        ],
        metadata={
            "scenario_type": current_scenario["type"],
            "step": scenario.step,
        },
        reward=0,
    )
    
    # Multi-turn conversation like 2048
    while True:
        # Present current task (like board state in 2048)
        task_prompt = f"Task: {current_scenario['task']}\nType: {current_scenario['type']}\nChoose the best tool:"
        
        trajectory.messages_and_choices.append(
            {"role": "user", "content": task_prompt}
        )
        
        try:
            # Get model response
            messages = trajectory.messages()
            chat_completion = await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
            )
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e
        
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)
        
        # Parse tool choice
        chosen_tool = "firecrawl_scrape"  # default
        try:
            if "<tool_choice>" in content and "</tool_choice>" in content:
                start = content.find("<tool_choice>") + len("<tool_choice>")
                end = content.find("</tool_choice>")
                parsed_tool = content[start:end].strip()
                if parsed_tool in AVAILABLE_TOOLS:
                    chosen_tool = parsed_tool
        except:
            pass
        
        # Calculate move score
        move_score = calculate_move_score(chosen_tool, current_scenario["optimal"], move_number)
        total_score += move_score
        move_number += 1
        
        # Check if we should continue (like game ending in 2048)
        if move_number >= 3 or move_score <= 0:  # End after 3 moves or bad move
            # Calculate final reward like 2048
            final_reward = calculate_final_reward(total_score, move_number)
            
            trajectory.metrics["total_score"] = total_score
            trajectory.metrics["moves_made"] = move_number
            trajectory.metrics["move_number"] = move_number
            
            trajectory.reward = final_reward
            print(f"📊 Step {scenario.step}: {move_number} moves, score: {total_score:.1f}, reward: {final_reward:.3f}")
            break
        
        # Continue to next task (like next move in 2048)
        if current_scenario["next_tasks"]:
            current_scenario = random.choice(current_scenario["next_tasks"])
        else:
            break
    
    return trajectory

# Initialize everything exactly like example
model = art.TrainableModel(
    name="agent-001",
    project="mcp-agent",
    base_model="OpenPipe/Qwen3-14B-Instruct",
)

backend = ServerlessBackend()

async def main():
    """Run training exactly like examplecode3.md"""
    print("🚀 FINAL TRAINING - Following examplecode3.md exactly")
    
    # Register model
    await model.register(backend)
    
    # Initialize weave like example
    weave.init(model.project, settings={"print_call_link": False})
    
    print(f"✅ Model: {model.name}")
    print(f"✅ Project: {model.project}")
    print(f"✅ Base model: OpenPipe/Qwen3-14B-Instruct")
    
    # Training loop exactly like examplecode3.md (lines 307-321)
    for i in range(await model.get_step(), 20):
        print(f"\n🔄 TRAINING STEP {i + 1}/20")
        
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, AgentScenario(step=i)) for _ in range(18))
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=18,
        )
        
        await model.delete_checkpoints('train/reward')
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        
        print(f"✅ Step {i + 1} complete")
    
    # Deploy model like example
    last_step = await model.get_step()
    deployed_inference_model_name = f"{model.get_inference_name()}:step{last_step}"
    print(f"🚀 step {last_step} deployed as {deployed_inference_model_name}")
    
    # Test deployed model like example
    print(f"\n🎬 TESTING DEPLOYED MODEL")
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    test_tasks = [
        ("Research AI safety developments", "research"),
        ("Find hotels in Tokyo", "travel"),
        ("Extract competitor data", "web_scraping")
    ]
    
    for task, task_type in test_tasks:
        print(f"\n🧪 TEST: {task}")
        
        messages = [
            {
                "role": "system",
                "content": f"You are an excellent MCP agent. Available tools: {', '.join(AVAILABLE_TOOLS)}\nChoose the best tool: <tool_choice>tool_name</tool_choice>",
            },
            {
                "role": "user",
                "content": f"Task: {task}\nType: {task_type}\nChoose the best tool:"
            }
        ]
        
        try:
            content = (await client.chat.completions.create(
                model=deployed_inference_model_name,
                messages=messages,
            )).choices[0].message.content
            
            # Parse tool choice
            tool = "unknown"
            if "<tool_choice>" in content and "</tool_choice>" in content:
                start = content.find("<tool_choice>") + len("<tool_choice>")
                end = content.find("</tool_choice>")
                tool = content[start:end].strip()
            
            print(f"   🔧 Tool: {tool}")
            print(f"   📝 Response: {content}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())