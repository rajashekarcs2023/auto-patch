"""
Conscious Context Engine - Main Training Script
Self-Evolving Agent that learns optimal context selection
"""
import os
import asyncio
import random
from dotenv import load_dotenv
from typing import Dict, List
import weave

import art
from art.serverless.backend import ServerlessBackend
from research_env import ResearchEnvironment, ResearchScenario
from context_engine import ContextManager
from evaluation import evaluate_with_ruler

# Load environment variables
load_dotenv()

# Verify required API keys
required_keys = ["WANDB_API_KEY", "FIRECRAWL_API_KEY", "OPENAI_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        print(f"Warning: {key} not set. Some features may not work.")

random.seed(42)

class ContextAwareTrajectory(art.Trajectory):
    """Enhanced trajectory that tracks context usage and efficiency"""
    context_used: List[str] = []
    context_metrics: Dict[str, float] = {}
    task_success_score: float = 0.0
    efficiency_score: float = 0.0


async def setup_model():
    """Initialize model with serverless backend"""
    print("🚀 Initializing Conscious Context Engine...")
    
    # Create model
    model = art.TrainableModel(
        name="conscious-context-agent-001",
        project="conscious-context-engine", 
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    
    # Initialize serverless backend
    backend = ServerlessBackend()
    
    # Register model
    await model.register(backend)
    print(f"✅ Model registered: {model.name}")
    
    return model, backend


@weave.op
async def rollout(model: art.Model, scenario: ResearchScenario) -> ContextAwareTrajectory:
    """Execute a research task with context-aware processing"""
    
    # Initialize environment
    env = ResearchEnvironment(os.getenv("FIRECRAWL_API_KEY", ""))
    task = scenario.task
    
    print(f"📋 Task: {task.question}")
    
    # Get optimized context using current policy
    selected_chunks, context_metrics = env.context_manager.get_optimized_context(task.task_type)
    
    print(f"🧠 Selected {len(selected_chunks)}/{len(env.context_manager.context_pool)} context chunks")
    print(f"📊 Context efficiency: {context_metrics['efficiency_score']:.2f}")
    
    # Build context string
    context_content = "\n\n".join([
        f"[{chunk.source.upper()}] {chunk.content}" 
        for chunk in selected_chunks
    ])
    
    # Create trajectory
    traj = ContextAwareTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "task_id": task.id,
            "step": scenario.step,
            "task_type": task.task_type
        }
    )
    
    # Store context usage
    traj.context_used = [chunk.id for chunk in selected_chunks]
    traj.context_metrics = context_metrics
    
    # System prompt with context awareness instruction
    system_prompt = f"""You are a research assistant with access to curated information. 
Your goal is to provide comprehensive, accurate answers using the available context efficiently.

AVAILABLE CONTEXT:
{context_content}

Instructions:
1. Use the provided context to answer questions accurately
2. Cite which sources ([WEB], [DOCS], [MEMORY]) you use
3. If context is insufficient, say so clearly
4. Be concise but comprehensive
"""

    user_prompt = f"Research Question: {task.question}\n\nPlease provide a detailed answer based on the available context."
    
    # Build conversation
    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Get model response  
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
        )
        
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            temperature=0.7,
            max_tokens=1000
        )
        
        response_content = response.choices[0].message.content
        traj.messages_and_choices.append(response.choices[0])
        
        # Evaluate research quality
        quality_scores = env.evaluate_research_quality(response_content, task)
        traj.task_success_score = quality_scores["overall"]
        
        # Calculate efficiency score
        efficiency_score = env.get_context_efficiency_score(context_metrics)
        traj.efficiency_score = efficiency_score
        
        # Combined reward: task success + efficiency bonus
        traj.reward = traj.task_success_score + (efficiency_score * 0.3)
        
        # Store detailed metrics
        traj.metrics.update({
            "task_success": traj.task_success_score,
            "efficiency": efficiency_score,
            "context_selected": len(selected_chunks),
            "context_available": len(env.context_manager.context_pool),
            "context_ratio": context_metrics["selection_ratio"],
            "relevance": quality_scores["relevance"],
            "detail": quality_scores["detail"],
            "source_diversity": quality_scores["source_diversity"]
        })
        
        # Update context manager with performance feedback
        env.context_manager.record_performance(
            used_chunks=selected_chunks,
            task_success=traj.task_success_score,
            context_metrics=context_metrics
        )
        
        print(f"✅ Task success: {traj.task_success_score:.2f}")
        print(f"⚡ Efficiency: {efficiency_score:.2f}")
        print(f"🎯 Combined reward: {traj.reward:.2f}")
        
    except Exception as e:
        print(f"❌ Error in rollout: {e}")
        traj.reward = -1.0
        traj.task_success_score = 0.0
        traj.efficiency_score = 0.0
    
    return traj


async def run_training_step(model: art.Model, step: int):
    """Run a single training step with multiple rollouts"""
    print(f"\n🏋️ Training Step {step}")
    print("=" * 50)
    
    # Create environment and get tasks
    env = ResearchEnvironment(os.getenv("FIRECRAWL_API_KEY", ""))
    tasks = env.get_research_tasks()
    
    # Create scenarios for this step
    scenarios = []
    for _ in range(2):  # 2 tasks per step
        task = random.choice(tasks)
        scenario = env.create_scenario(step=step, task_id=task.id)
        scenarios.append(scenario)
    
    # Generate trajectory groups (multiple attempts per scenario for comparison)
    groups = []
    for scenario in scenarios:
        group = art.TrajectoryGroup(
            trajectories=[
                rollout(model, scenario) 
                for _ in range(3)  # 3 attempts per scenario
            ]
        )
        groups.append(group)
    
    # Gather all trajectory groups
    print("🔄 Gathering trajectories...")
    finished_groups = await art.gather_trajectory_groups(
        groups,
        pbar_desc=f"Step {step}",
        max_exceptions=6  # 2 scenarios * 3 attempts
    )
    
    # Use RULER to score trajectories within each group
    print("🏆 Evaluating with RULER...")
    judged_groups = []
    for group in finished_groups:
        try:
            judged_group = await evaluate_with_ruler(group)
            judged_groups.append(judged_group)
        except Exception as e:
            print(f"⚠️ RULER evaluation failed: {e}")
            # Use fallback scoring
            for traj in group.trajectories:
                if hasattr(traj, 'reward') and traj.reward > 0:
                    pass  # Keep existing reward
                else:
                    traj.reward = 0.0
            judged_groups.append(group)
    
    # Train the model
    print("🧠 Training model...")
    await model.delete_checkpoints()
    await model.train(
        judged_groups,
        config=art.TrainConfig(learning_rate=1e-5)
    )
    
    # Show evolution summary
    evolution = env.context_manager.get_evolution_summary()
    print(f"\n📈 Evolution Summary:")
    print(f"   Strategy: {evolution['current_strategy']}")
    print(f"   Learned chunks: {evolution['learned_chunk_weights']}")
    print(f"   Context efficiency: {evolution['context_efficiency']:.2f}")
    print(f"   Evolution stage: {evolution['evolution_stage']}")


async def run_demo():
    """Run a demo showing before/after evolution"""
    print("\n🎭 DEMO: Conscious Context Engine Evolution")
    print("=" * 60)
    
    env = ResearchEnvironment(os.getenv("FIRECRAWL_API_KEY", ""))
    
    # Reset to baseline strategy
    env.context_manager.policy.selection_strategy = "full"
    print("🔄 Reset to baseline (using full context)...")
    
    # Demo task
    demo_task = env.get_research_tasks()[0]  # AV regulations
    scenario = env.create_scenario(step=999, task_id=demo_task.id)
    
    print(f"\n📋 Demo Task: {demo_task.question}")
    
    # Show baseline performance
    selected_chunks, metrics = env.context_manager.get_optimized_context(demo_task.task_type)
    print(f"\n📊 BASELINE:")
    print(f"   Context chunks used: {len(selected_chunks)}/{len(env.context_manager.context_pool)}")
    print(f"   Selection ratio: {metrics['selection_ratio']:.2f}")
    print(f"   Efficiency score: {metrics['efficiency_score']:.2f}")
    
    # Switch to learned strategy
    env.context_manager.policy.selection_strategy = "learned"
    selected_chunks, metrics = env.context_manager.get_optimized_context(demo_task.task_type)
    print(f"\n📊 EVOLVED:")
    print(f"   Context chunks used: {len(selected_chunks)}/{len(env.context_manager.context_pool)}")
    print(f"   Selection ratio: {metrics['selection_ratio']:.2f}")
    print(f"   Efficiency score: {metrics['efficiency_score']:.2f}")
    
    improvement = (1 - metrics['selection_ratio']) * 100
    print(f"\n🎯 IMPROVEMENT: {improvement:.1f}% context reduction!")


async def main():
    """Main training loop"""
    print("🧠 Conscious Context Engine - Self-Evolving AI")
    print("=" * 60)
    
    # Initialize Weave for tracking
    if os.getenv("WANDB_API_KEY"):
        weave.init("conscious-context-engine", settings={"print_call_link": False})
    
    # Setup model
    model, backend = await setup_model()
    
    # Training configuration
    max_steps = 5  # Keep it short for demo
    current_step = await model.get_step()
    
    print(f"\n🏁 Starting training from step {current_step}")
    print(f"🎯 Target: {max_steps} steps")
    
    # Training loop
    for step in range(current_step, max_steps):
        try:
            await run_training_step(model, step)
            
            # Show demo every 2 steps
            if step % 2 == 1:
                await run_demo()
                
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in step {step}: {e}")
            continue
    
    print(f"\n🎉 Training completed! Final step: {await model.get_step()}")
    
    # Final demo
    await run_demo()


if __name__ == "__main__":
    asyncio.run(main())