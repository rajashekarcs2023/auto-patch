#!/usr/bin/env python3
"""
Test the EXACT training call to reproduce the exceptions
"""
import asyncio
import traceback
import random
import requests
import weave
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

import art
from art.serverless.backend import ServerlessBackend

load_dotenv()
random.seed(42)

class AgentScenario(BaseModel):
    step: int

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def exact_rollout(model: art.Model, scenario: AgentScenario) -> art.Trajectory:
    """EXACT copy of our training rollout function"""
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Simplified version to isolate the issue
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are a test agent. Respond with: <tool_choice>perplexity_search</tool_choice>"
            }
        ],
        metadata={"step": scenario.step},
        reward=0,
    )
    
    # Add user message
    trajectory.messages_and_choices.append({
        "role": "user", 
        "content": "Choose a tool"
    })
    
    try:
        print(f"  🔄 Making call for scenario step {scenario.step}...")
        
        # Get messages
        messages = trajectory.messages()
        
        # Make LLM call - EXACT same parameters as training
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=128,
            messages=messages,
            model=model.get_inference_name(),
        )
        
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        
        # Add response to trajectory
        trajectory.messages_and_choices.append(choice)
        
        # Set simple reward
        trajectory.reward = 0.5 + random.uniform(-0.1, 0.1)
        
        # Add numeric metrics only
        trajectory.metrics.update({
            "test_metric": float(random.uniform(0, 1)),
            "step": float(scenario.step)
        })
        
        print(f"  ✅ Success for step {scenario.step}, reward: {trajectory.reward:.3f}")
        return trajectory
        
    except Exception as e:
        print(f"  ❌ Failed for step {scenario.step}: {type(e).__name__}: {e}")
        trajectory.reward = -1.0
        trajectory.metrics["error"] = 1.0
        raise e  # Re-raise to trigger art.retry

async def test_exact_gather():
    """Test the exact art.gather_trajectory_groups call"""
    print("🧪 TESTING EXACT TRAINING GATHER")
    print("=" * 50)
    
    # Initialize model exactly like training
    weave.init("test-exact", settings={"print_call_link": False})
    
    model = art.TrainableModel(
        name="agent-001",
        project="mcp-agent",
        base_model="OpenPipe/Qwen3-14B-Instruct"
    )
    
    backend = ServerlessBackend()
    await model.register(backend)
    
    step = await model.get_step()
    print(f"✅ Model at step: {step}")
    
    # Create scenarios exactly like training
    scenarios = [AgentScenario(step=0) for _ in range(18)]
    
    try:
        print(f"🔄 Running art.gather_trajectory_groups with 18 rollouts...")
        
        # EXACT same call as training
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(exact_rollout(model, scenario) for scenario in scenarios)
                for _ in range(1)
            ),
            pbar_desc="test gather",
            max_exceptions=18,
        )
        
        print(f"✅ Gather completed successfully!")
        print(f"Groups returned: {len(train_groups)}")
        
        # Check results
        for i, group in enumerate(train_groups):
            print(f"Group {i}: {len(group.trajectories)} trajectories")
            for j, traj in enumerate(group.trajectories):
                print(f"  Traj {j}: reward={traj.reward:.3f}, metrics={traj.metrics}")
        
    except Exception as e:
        print(f"❌ Gather failed: {type(e).__name__}: {e}")
        traceback.print_exc()

async def test_training_step():
    """Test a complete training step"""
    print(f"\n🧪 TESTING COMPLETE TRAINING STEP")
    print("=" * 50)
    
    weave.init("test-step", settings={"print_call_link": False})
    
    model = art.TrainableModel(
        name="agent-001",
        project="mcp-agent",
        base_model="OpenPipe/Qwen3-14B-Instruct"
    )
    
    backend = ServerlessBackend()
    await model.register(backend)
    
    try:
        scenarios = [AgentScenario(step=0) for _ in range(6)]  # Smaller test
        
        print(f"🔄 Running complete training step...")
        
        # Gather trajectories
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(exact_rollout(model, scenario) for scenario in scenarios)
                for _ in range(1)
            ),
            pbar_desc="test step",
            max_exceptions=6,
        )
        
        print(f"✅ Trajectory gathering successful")
        
        # Try training
        await model.delete_checkpoints('train/reward')
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5)
        )
        
        print(f"✅ Training step completed successfully!")
        
    except Exception as e:
        print(f"❌ Training step failed: {type(e).__name__}: {e}")
        traceback.print_exc()

async def main():
    """Run exact training reproduction tests"""
    print("🧪 REPRODUCING EXACT TRAINING FAILURES")
    print("=" * 60)
    
    await test_exact_gather()
    await test_training_step()
    
    print(f"\n🏁 EXACT TRAINING TEST COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())