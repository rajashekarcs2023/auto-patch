"""
RULER-based evaluation for context-aware research tasks
"""
import os
from typing import List
import art
from art.rewards import ruler_score_group


async def evaluate_with_ruler(trajectory_group: art.TrajectoryGroup, model: str = "openai/gpt-4") -> art.TrajectoryGroup:
    """
    Use RULER to evaluate and score a group of trajectories
    
    RULER will compare different approaches to the same task and assign
    relative scores based on which responses are better.
    """
    
    # Ensure we have OpenAI API key for RULER
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ No OpenAI API key found. Using fallback scoring.")
        return trajectory_group
    
    try:
        # Use RULER to score the group
        judged_group = await ruler_score_group(
            trajectory_group, 
            model=model,
            debug=False  # Set to True for debugging
        )
        
        if judged_group is None:
            print("⚠️ RULER returned None, using original group")
            return trajectory_group
            
        return judged_group
        
    except Exception as e:
        print(f"⚠️ RULER evaluation failed: {e}")
        print("   Using fallback reward calculation...")
        
        # Fallback: use the reward we calculated in rollout
        for traj in trajectory_group.trajectories:
            if not hasattr(traj, 'reward') or traj.reward is None:
                traj.reward = 0.0
        
        return trajectory_group


def create_evaluation_prompt(task_question: str, context_used: List[str]) -> str:
    """
    Create a prompt for LLM-based evaluation of research quality
    """
    context_summary = f"Used {len(context_used)} context chunks"
    
    return f"""
Evaluate this research response on the following criteria:

Task: {task_question}
Context Information: {context_summary}

Rate the response on:
1. Accuracy and relevance to the question (0-1)
2. Use of provided context sources (0-1) 
3. Comprehensiveness of the answer (0-1)
4. Efficiency of context usage (0-1)

Return a score from 0.0 to 1.0 representing overall quality.
"""


async def simple_llm_evaluation(research_output: str, task_question: str, context_used: List[str]) -> float:
    """
    Simple LLM-based evaluation as backup to RULER
    """
    try:
        from litellm import acompletion
        
        evaluation_prompt = create_evaluation_prompt(task_question, context_used)
        
        messages = [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": f"Research Output: {research_output}"}
        ]
        
        response = await acompletion(
            model="openai/gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        # Extract numeric score from response
        content = response.choices[0].message.content.lower()
        
        # Simple extraction - look for decimal number
        import re
        scores = re.findall(r'0\.\d+|1\.0|[01]', content)
        if scores:
            return float(scores[0])
        else:
            return 0.5  # Default middle score
            
    except Exception as e:
        print(f"LLM evaluation failed: {e}")
        return 0.5  # Default score


def calculate_context_efficiency_reward(selected_chunks: int, total_chunks: int, task_success: float) -> float:
    """
    Calculate reward based on context efficiency
    
    Rewards agents for using fewer context chunks while maintaining quality
    """
    if total_chunks == 0:
        return 0.0
    
    selection_ratio = selected_chunks / total_chunks
    
    # Efficiency bonus: reward for using less context
    efficiency = 1.0 - selection_ratio
    
    # Only give efficiency bonus if task was successful
    efficiency_reward = efficiency * task_success * 0.5
    
    return efficiency_reward


def get_evaluation_metrics(trajectory) -> dict:
    """
    Extract evaluation metrics from a trajectory
    """
    metrics = {}
    
    if hasattr(trajectory, 'metrics'):
        metrics.update(trajectory.metrics)
    
    if hasattr(trajectory, 'task_success_score'):
        metrics['task_success'] = trajectory.task_success_score
    
    if hasattr(trajectory, 'efficiency_score'):
        metrics['efficiency'] = trajectory.efficiency_score
    
    if hasattr(trajectory, 'context_used'):
        metrics['context_chunks_used'] = len(trajectory.context_used)
    
    return metrics