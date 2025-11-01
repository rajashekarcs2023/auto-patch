"""
Core Context Selection Engine - The Self-Evolving Brain
"""
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class ContextChunk:
    """Individual piece of context/memory"""
    id: str
    content: str
    source: str  # "web", "memory", "docs"
    relevance_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass 
class ContextPolicy:
    """The agent's learned policy for context selection"""
    chunk_weights: Dict[str, float] = field(default_factory=dict)
    source_preferences: Dict[str, float] = field(default_factory=lambda: {
        "web": 1.0, "memory": 1.0, "docs": 1.0
    })
    max_context_length: int = 2000  # Learned optimal length
    selection_strategy: str = "learned"  # "full", "recent", "learned"
    
    def select_context(self, available_chunks: List[ContextChunk], task_type: str) -> List[ContextChunk]:
        """Select optimal context based on learned policy"""
        if self.selection_strategy == "full":
            return available_chunks
        elif self.selection_strategy == "recent":
            return available_chunks[-5:]  # Last 5 chunks
        
        # Learned selection strategy
        scored_chunks = []
        for chunk in available_chunks:
            # Calculate score based on learned weights
            base_score = self.chunk_weights.get(chunk.id, 0.5)
            source_boost = self.source_preferences.get(chunk.source, 1.0)
            success_boost = chunk.success_rate
            
            final_score = base_score * source_boost * (1 + success_boost)
            scored_chunks.append((chunk, final_score))
        
        # Sort by score and take top chunks within length limit
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        total_length = 0
        for chunk, score in scored_chunks:
            if total_length + len(chunk.content) <= self.max_context_length:
                selected.append(chunk)
                total_length += len(chunk.content)
            else:
                break
                
        return selected
    
    def update_from_reward(self, used_chunks: List[ContextChunk], task_success: float, efficiency_bonus: float):
        """Update policy based on performance feedback"""
        for chunk in used_chunks:
            # Update chunk-specific weights
            old_weight = self.chunk_weights.get(chunk.id, 0.5)
            reward_signal = task_success + efficiency_bonus
            
            # Simple learning rate of 0.1
            new_weight = old_weight + 0.1 * (reward_signal - old_weight)
            self.chunk_weights[chunk.id] = max(0.0, min(1.0, new_weight))
            
            # Update chunk success tracking
            chunk.usage_count += 1
            chunk.success_rate = ((chunk.success_rate * (chunk.usage_count - 1)) + task_success) / chunk.usage_count
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Return metrics showing how policy has evolved"""
        total_chunks = len(self.chunk_weights)
        high_value_chunks = sum(1 for w in self.chunk_weights.values() if w > 0.7)
        
        return {
            "total_chunks_learned": total_chunks,
            "high_value_chunk_ratio": high_value_chunks / max(total_chunks, 1),
            "max_context_efficiency": 1.0 - (self.max_context_length / 5000),  # Assuming 5000 was original max
            "strategy_evolution": 1.0 if self.selection_strategy == "learned" else 0.0
        }


class ContextManager:
    """Manages the pool of available context and the selection policy"""
    
    def __init__(self):
        self.context_pool: List[ContextChunk] = []
        self.policy = ContextPolicy()
        self.memory_file = "context_memory.json"
        self.load_policy()
    
    def add_context(self, content: str, source: str, chunk_id: Optional[str] = None) -> str:
        """Add new context to the pool"""
        if chunk_id is None:
            chunk_id = f"{source}_{len(self.context_pool)}"
        
        chunk = ContextChunk(
            id=chunk_id,
            content=content,
            source=source
        )
        self.context_pool.append(chunk)
        return chunk_id
    
    def get_optimized_context(self, task_type: str) -> Tuple[List[ContextChunk], Dict[str, float]]:
        """Get context optimized by current policy + return metrics"""
        selected_chunks = self.policy.select_context(self.context_pool, task_type)
        
        metrics = {
            "total_available": len(self.context_pool),
            "selected_count": len(selected_chunks),
            "selection_ratio": len(selected_chunks) / max(len(self.context_pool), 1),
            "total_selected_length": sum(len(c.content) for c in selected_chunks),
            "efficiency_score": self.policy.get_efficiency_metrics()["max_context_efficiency"]
        }
        
        return selected_chunks, metrics
    
    def record_performance(self, used_chunks: List[ContextChunk], task_success: float, context_metrics: Dict[str, float]):
        """Record how well the selected context performed"""
        # Calculate efficiency bonus based on context selection
        efficiency_bonus = context_metrics.get("efficiency_score", 0.0)
        
        # Update policy
        self.policy.update_from_reward(used_chunks, task_success, efficiency_bonus)
        
        # Evolve strategy if we're getting good results
        if task_success > 0.8 and efficiency_bonus > 0.5:
            self.policy.selection_strategy = "learned"
        
        self.save_policy()
    
    def save_policy(self):
        """Save learned policy to disk"""
        policy_data = {
            "chunk_weights": self.policy.chunk_weights,
            "source_preferences": self.policy.source_preferences,
            "max_context_length": self.policy.max_context_length,
            "selection_strategy": self.policy.selection_strategy
        }
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(policy_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save policy: {e}")
    
    def load_policy(self):
        """Load previously learned policy"""
        try:
            with open(self.memory_file, 'r') as f:
                policy_data = json.load(f)
                self.policy.chunk_weights = policy_data.get("chunk_weights", {})
                self.policy.source_preferences = policy_data.get("source_preferences", {
                    "web": 1.0, "memory": 1.0, "docs": 1.0
                })
                self.policy.max_context_length = policy_data.get("max_context_length", 2000)
                self.policy.selection_strategy = policy_data.get("selection_strategy", "learned")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No previous policy found, starting fresh")
    
    def get_evolution_summary(self) -> Dict[str, any]:
        """Get summary of how the agent has evolved"""
        metrics = self.policy.get_efficiency_metrics()
        
        return {
            "total_context_chunks": len(self.context_pool),
            "learned_chunk_weights": len(self.policy.chunk_weights),
            "current_strategy": self.policy.selection_strategy,
            "context_efficiency": metrics["max_context_efficiency"],
            "high_value_chunks": metrics["high_value_chunk_ratio"],
            "evolution_stage": "evolved" if metrics["strategy_evolution"] > 0 else "baseline"
        }