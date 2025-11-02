"""
Self-Improving Reasoning Chain Engine - Main Training
Revolutionary demonstration of AI that learns to think better
"""
import os
import asyncio
import random
from dotenv import load_dotenv
from typing import Dict, List
import weave

import art
from art.serverless.backend import ServerlessBackend
from reasoning_engine import MetaReasoningEngine, ReasoningTaskEnvironment, ReasoningChain
from evaluation import evaluate_with_ruler
from task_benchmarks import TaskPerformanceTracker

load_dotenv()

class ReasoningTrajectory(art.Trajectory):
    """Enhanced trajectory tracking reasoning evolution"""
    reasoning_chain_used: str = ""
    reasoning_steps_quality: List[float] = []
    reasoning_evolution_step: bool = False
    problem_type: str = ""
    baseline_comparison: float = 0.0


async def setup_reasoning_model():
    """Initialize model for reasoning improvement"""
    print("Initializing Self-Improving Reasoning Engine...")
    
    model = art.TrainableModel(
        name="reasoning-evolution-agent-001",
        project="self-improving-reasoning",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    
    backend = ServerlessBackend()
    await model.register(backend)
    
    print(f"Model registered: {model.name}")
    return model, backend


@weave.op
async def reasoning_rollout(model: art.Model, problem: Dict, reasoning_engine: MetaReasoningEngine, step: int) -> ReasoningTrajectory:
    """Execute problem solving with current reasoning chain"""
    
    problem_type = problem["type"]
    problem_statement = problem["problem"]
    
    print(f"Problem: {problem['id']} ({problem_type})")
    
    # Get current best reasoning chain for this problem type
    reasoning_chain = reasoning_engine.get_reasoning_chain(problem_type)
    
    # Store chain performance for tracking
    chain_performance = reasoning_chain.overall_success_rate
    
    # Create reasoning-guided prompt
    reasoning_template = reasoning_chain.to_prompt_template()
    
    system_prompt = f"""You are an advanced AI that follows systematic reasoning patterns to solve complex problems.

For this problem, use the following reasoning approach:

{reasoning_template}

Be thorough and systematic in each step. Show your thinking clearly."""

    user_prompt = f"""Problem to solve: {problem_statement}

Please follow the reasoning pattern step by step and provide a comprehensive solution."""

    # Create trajectory
    traj = ReasoningTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "problem_id": problem["id"],
            "problem_type": problem_type,
            "reasoning_chain_id": reasoning_chain.chain_id,
            "step": step
        }
    )
    
    traj.reasoning_chain_used = reasoning_chain.chain_id
    traj.problem_type = problem_type
    
    # Build conversation
    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
        )
        
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=traj.messages(),
            temperature=0.7,
            max_tokens=1500
        )
        
        response_content = response.choices[0].message.content
        traj.messages_and_choices.append(response.choices[0])
        
        # Evaluate reasoning quality using passed reasoning_engine
        from reasoning_engine import ReasoningTaskEnvironment
        env = ReasoningTaskEnvironment()
        env.meta_engine = reasoning_engine  # Use shared engine
        reasoning_score, step_results = env.evaluate_reasoning_quality(response_content, problem)
        
        # Record reasoning performance
        reasoning_engine.record_reasoning_outcome(
            chain_id=reasoning_chain.chain_id,
            step_results=step_results,
            overall_success=reasoning_score > 0.7,
            task_type=problem_type
        )
        
        # Calculate reward (reasoning quality + improvement bonus)
        traj.reward = reasoning_score
        
        # Check if this triggered reasoning evolution
        evolved_chain = reasoning_engine.evolve_reasoning_chain(
            reasoning_chain.chain_id, 
            problem_type
        )
        
        if evolved_chain:
            traj.reasoning_evolution_step = True
            traj.reward += 0.3  # Bonus for triggering evolution
            print(f"REASONING EVOLVED: {reasoning_chain.chain_id} -> {evolved_chain.chain_id}")
        
        # Update chain performance after recording outcome
        updated_chain_performance = reasoning_chain.overall_success_rate
        
        # Store detailed metrics
        traj.metrics.update({
            "reasoning_score": reasoning_score,
            "chain_performance": updated_chain_performance,
            "evolution_triggered": traj.reasoning_evolution_step,
            "step_success_count": sum(1 for _, success in step_results if success),
            "total_reasoning_steps": len(step_results)
        })
        
        traj.reasoning_steps_quality = [score for _, score in step_results]
        
        print(f"Reasoning Score: {reasoning_score:.2f}")
        print(f"Chain Performance: {updated_chain_performance:.2f}")
        if evolved_chain:
            print(f"Evolution Triggered: New chain {evolved_chain.chain_id}")
        
    except Exception as e:
        print(f"Error in reasoning rollout: {e}")
        traj.reward = -1.0
    
    return traj


async def run_reasoning_training_step(model: art.Model, reasoning_engine: MetaReasoningEngine, step: int):
    """Run training step focused on reasoning improvement"""
    print(f"\nReasoning Training Step {step}")
    print("=" * 50)
    
    # Get test problems
    env = ReasoningTaskEnvironment()
    problems = env.get_test_problems()
    
    # Select problems for this step (mix of types)
    selected_problems = random.sample(problems, min(3, len(problems)))
    
    # Generate trajectory groups
    groups = []
    for problem in selected_problems:
        group = art.TrajectoryGroup(
            trajectories=[
                reasoning_rollout(model, problem, reasoning_engine, step)
                for _ in range(4)  # 4 attempts per problem for comparison
            ]
        )
        groups.append(group)
    
    print("Gathering reasoning trajectories...")
    finished_groups = await art.gather_trajectory_groups(
        groups,
        pbar_desc=f"Reasoning Step {step}",
        max_exceptions=12  # 3 problems * 4 attempts
    )
    
    # Evaluate with RULER
    print("Evaluating reasoning quality...")
    judged_groups = []
    for group in finished_groups:
        try:
            judged_group = await evaluate_with_ruler(group)
            judged_groups.append(judged_group)
        except Exception as e:
            print(f"RULER evaluation failed: {e}")
            # Use existing rewards
            judged_groups.append(group)
    
    # Train the model
    print("Training model on reasoning improvements...")
    await model.delete_checkpoints()
    await model.train(
        judged_groups,
        config=art.TrainConfig(learning_rate=1e-5)
    )
    
    # Show reasoning evolution metrics
    evolution_metrics = reasoning_engine.get_reasoning_evolution_metrics()
    print(f"\nReasoning Evolution Metrics:")
    print(f"   Total Chains: {evolution_metrics['total_reasoning_chains']}")
    print(f"   Evolved Chains: {evolution_metrics['evolved_chains']}")
    print(f"   Average Performance: {evolution_metrics['average_chain_performance']:.2f}")
    print(f"   Reasoning Sophistication: {evolution_metrics['reasoning_sophistication']:.2f}")
    
    # Save reasoning memory
    reasoning_engine.save_reasoning_memory()


async def demonstrate_reasoning_evolution():
    """Demonstrate before/after reasoning improvement"""
    print("\nREASONING EVOLUTION DEMONSTRATION")
    print("=" * 60)
    
    reasoning_engine = MetaReasoningEngine()
    env = ReasoningTaskEnvironment()
    
    # Get a test problem
    test_problem = env.get_test_problems()[0]  # Technical OAuth problem
    
    print(f"Test Problem: {test_problem['problem']}")
    print()
    
    # Show baseline reasoning chain
    print("BASELINE REASONING PATTERN:")
    baseline_chain = reasoning_engine.get_reasoning_chain(test_problem["type"])
    print(baseline_chain.to_prompt_template())
    print(f"Performance: {baseline_chain.overall_success_rate:.2f}")
    print()
    
    # Simulate some learning (mock poor performance to trigger evolution)
    print("SIMULATING LEARNING PROCESS...")
    for i in range(10):
        reasoning_engine.record_reasoning_outcome(
            chain_id=baseline_chain.chain_id,
            step_results=[(0, False), (1, True), (2, False), (3, True)],  # Mixed results
            overall_success=False,  # Poor performance
            task_type=test_problem["type"]
        )
    
    # Trigger evolution
    evolved_chain = reasoning_engine.evolve_reasoning_chain(
        baseline_chain.chain_id, 
        test_problem["type"]
    )
    
    if evolved_chain:
        print(f"REASONING EVOLVED: {baseline_chain.chain_id} -> {evolved_chain.chain_id}")
        print("\nEVOLVED REASONING PATTERN:")
        print(evolved_chain.to_prompt_template())
        print()
        
        print("KEY IMPROVEMENTS:")
        print("• More systematic analysis approach")
        print("• Enhanced hypothesis generation")
        print("• Better verification methods")
        print("• Meta-reasoning for complex problems")
        print()
    
    # Show evolution metrics
    metrics = reasoning_engine.get_reasoning_evolution_metrics()
    print("EVOLUTION METRICS:")
    print(f"   Reasoning Sophistication: {metrics['reasoning_sophistication']:.2f}")
    print(f"   Evolution Rate: {metrics['evolution_rate']:.2f}")
    print(f"   Step Effectiveness: {metrics['step_effectiveness']}")


async def main():
    """Main reasoning improvement training"""
    print("Self-Improving Reasoning Chain Engine")
    print("=" * 60)
    print("Revolutionary AI that learns to rewrite its own reasoning process")
    print()
    
    # Initialize Weave
    if os.getenv("WANDB_API_KEY"):
        weave.init("self-improving-reasoning", settings={"print_call_link": False})
    
    # Quick demonstration
    await demonstrate_reasoning_evolution()
    
    # Setup model
    model, backend = await setup_reasoning_model()
    reasoning_engine = MetaReasoningEngine()
    
    # Training configuration (lightweight for demo)
    max_steps = 8
    current_step = await model.get_step()
    
    print(f"\nStarting reasoning training from step {current_step}")
    
    # Initialize task performance tracker
    task_tracker = TaskPerformanceTracker()
    baseline_results = None
    
    # Training loop
    for step in range(current_step, max_steps):
        try:
            await run_reasoning_training_step(model, reasoning_engine, step)
            
            # Run task benchmarks every 2 steps
            if step % 2 == 1:
                print(f"\n🎯 RUNNING TASK BENCHMARKS (Step {step})")
                benchmark_results = await task_tracker.run_comprehensive_benchmark(model, reasoning_engine)
                
                # Save results
                filename = task_tracker.save_benchmark_results(benchmark_results)
                
                # Compare with baseline if available
                if baseline_results is None:
                    baseline_results = benchmark_results
                    print("\n📊 BASELINE ESTABLISHED")
                    # Save baseline for comparison
                    baseline_filename = filename.replace("task_benchmark_results_", "task_benchmark_results_baseline_")
                    task_tracker.save_benchmark_results(baseline_results, baseline_filename)
                else:
                    comparison = task_tracker.compare_benchmark_runs(
                        filename.replace("task_benchmark_results_", "task_benchmark_results_baseline_"), 
                        benchmark_results
                    )
                    if comparison.get("summary"):
                        improvement = comparison["summary"].get("improvement_percent", 0)
                        print(f"\n📈 IMPROVEMENT: {improvement:.1f}% vs baseline")
                
                await demonstrate_reasoning_evolution()
                
        except KeyboardInterrupt:
            print("\nTraining interrupted")
            break
        except Exception as e:
            print(f"Error in step {step}: {e}")
            continue
    
    print(f"\nReasoning training completed! Final step: {await model.get_step()}")
    
    # Final comprehensive benchmark and summary
    print("\n" + "="*60)
    print("FINAL PERFORMANCE EVALUATION")
    print("="*60)
    
    final_results = await task_tracker.run_comprehensive_benchmark(model, reasoning_engine)
    final_filename = task_tracker.save_benchmark_results(final_results, "final_benchmark_results.json")
    
    if baseline_results:
        print("\n🔍 TRAINING IMPACT ANALYSIS")
        print("-" * 40)
        final_comparison = task_tracker.compare_benchmark_runs("task_benchmark_results_baseline_*.json", final_results)
        
        if final_comparison.get("summary"):
            total_improvement = final_comparison["summary"].get("improvement_percent", 0)
            print(f"📊 OVERALL IMPROVEMENT: {total_improvement:.1f}%")
            print(f"📈 TASKS IMPROVED: {final_comparison['summary']['tasks_improved']}")
            print(f"🚀 SIGNIFICANT IMPROVEMENTS: {final_comparison['summary']['significant_improvements']}")
    
    # Show reasoning evolution summary
    evolution_metrics = reasoning_engine.get_reasoning_evolution_metrics()
    print(f"\n🧠 REASONING EVOLUTION SUMMARY:")
    print(f"   Total Chains Created: {evolution_metrics['total_reasoning_chains']}")
    print(f"   Chains Evolved: {evolution_metrics['evolved_chains']}")
    print(f"   Evolution Rate: {evolution_metrics['evolution_rate']:.1%}")
    print(f"   Reasoning Sophistication: {evolution_metrics['reasoning_sophistication']:.3f}")
    
    # Final demonstration
    await demonstrate_reasoning_evolution()


if __name__ == "__main__":
    asyncio.run(main())