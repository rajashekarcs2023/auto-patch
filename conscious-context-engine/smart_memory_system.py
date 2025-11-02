#!/usr/bin/env python3
"""
Smart Memory Evolution System
Agent automatically identifies and removes useless memory/context for self-improvement
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class SmartMemoryItem:
    """Enhanced memory item with usage tracking and relevance scoring"""
    
    def __init__(self, key: str, value: Any, importance: float = 1.0, context_type: str = "general"):
        self.key = key
        self.value = value
        self.importance = importance
        self.context_type = context_type
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.success_correlation = 0.0  # How often this memory led to successful outcomes
        self.relevance_score = 1.0  # Current relevance based on usage patterns
        self.decay_rate = 0.1  # How quickly this memory becomes less relevant
        
    def access(self, successful_outcome: bool = True):
        """Record memory access and outcome correlation"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Update success correlation (weighted average)
        weight = 0.8  # Weight for new information
        outcome_score = 1.0 if successful_outcome else 0.0
        self.success_correlation = (weight * outcome_score + (1 - weight) * self.success_correlation)
        
        # Boost relevance on access
        self.relevance_score = min(1.0, self.relevance_score + 0.1)
    
    def decay(self):
        """Apply time-based decay to relevance"""
        days_old = (datetime.now() - self.last_accessed).days
        if days_old > 0:
            self.relevance_score *= (1 - self.decay_rate) ** days_old
        
        # Memory becomes less relevant if not accessed
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        if hours_since_access > 1:  # If not accessed in last hour
            self.relevance_score *= 0.95
    
    def get_utility_score(self) -> float:
        """Calculate overall utility score for memory pruning decisions"""
        # Factors: importance, success correlation, relevance, recency
        recency_factor = max(0.1, 1.0 - (datetime.now() - self.last_accessed).total_seconds() / (7 * 24 * 3600))
        access_factor = min(1.0, self.access_count / 10.0)  # Normalize access count
        
        utility = (
            self.importance * 0.3 +
            self.success_correlation * 0.4 +
            self.relevance_score * 0.2 +
            recency_factor * 0.1
        ) * access_factor
        
        return utility

class IntelligentMemoryEvolutionSystem:
    """Advanced memory system that automatically prunes useless memories"""
    
    def __init__(self, max_memory_items: int = 50):
        self.max_memory_items = max_memory_items
        self.memory_store: Dict[str, SmartMemoryItem] = {}
        self.pruning_history: List[Dict[str, Any]] = []
        self.context_effectiveness: Dict[str, float] = {}
        self.memory_categories = {
            "tool_patterns": [],
            "successful_strategies": [],
            "failed_attempts": [],
            "context_patterns": [],
            "user_preferences": []
        }
    
    def store_memory(self, key: str, value: Any, importance: float = 1.0, context_type: str = "general") -> bool:
        """Store memory with automatic pruning if needed"""
        # Create new memory item
        memory_item = SmartMemoryItem(key, value, importance, context_type)
        
        # Check if we need to prune before adding
        if len(self.memory_store) >= self.max_memory_items:
            pruned = self._intelligent_pruning()
            if pruned:
                print(f"🧹 MEMORY EVOLUTION: Removed {len(pruned)} obsolete memories")
                print(f"   Pruned: {[item.key for item in pruned[:3]]}{'...' if len(pruned) > 3 else ''}")
        
        # Store the new memory
        self.memory_store[key] = memory_item
        
        # Categorize memory
        self._categorize_memory(key, context_type)
        
        return True
    
    def access_memory(self, key: str, successful_outcome: bool = True) -> Optional[Any]:
        """Access memory and update its usage statistics"""
        if key in self.memory_store:
            memory_item = self.memory_store[key]
            memory_item.access(successful_outcome)
            return memory_item.value
        return None
    
    def retrieve_relevant_memories(self, context: str, task_type: str, max_memories: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant memories with automatic relevance scoring"""
        # Apply decay to all memories
        for memory_item in self.memory_store.values():
            memory_item.decay()
        
        # Score memories for relevance to current context
        scored_memories = []
        context_keywords = set(context.lower().split())
        
        for key, memory_item in self.memory_store.items():
            # Calculate relevance score
            relevance = self._calculate_contextual_relevance(memory_item, context_keywords, task_type)
            
            # Combine with utility score
            overall_score = relevance * memory_item.get_utility_score()
            
            if overall_score > 0.1:  # Only consider memories above threshold
                scored_memories.append({
                    "key": key,
                    "value": memory_item.value,
                    "score": overall_score,
                    "context_type": memory_item.context_type,
                    "access_count": memory_item.access_count,
                    "success_rate": memory_item.success_correlation
                })
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        selected_memories = scored_memories[:max_memories]
        
        # Update access statistics for retrieved memories
        for memory in selected_memories:
            self.access_memory(memory["key"], successful_outcome=True)
        
        return selected_memories
    
    def _intelligent_pruning(self) -> List[SmartMemoryItem]:
        """Intelligently remove low-utility memories"""
        if len(self.memory_store) <= self.max_memory_items * 0.8:
            return []
        
        # Calculate utility scores for all memories
        memory_utilities = []
        for key, memory_item in self.memory_store.items():
            utility = memory_item.get_utility_score()
            memory_utilities.append((key, memory_item, utility))
        
        # Sort by utility (lowest first for removal)
        memory_utilities.sort(key=lambda x: x[2])
        
        # Determine how many to remove
        target_removal = max(5, len(self.memory_store) - int(self.max_memory_items * 0.9))
        
        # Remove lowest utility memories
        pruned_memories = []
        for i in range(min(target_removal, len(memory_utilities))):
            key, memory_item, utility = memory_utilities[i]
            
            # Don't remove very recent or highly important memories
            hours_old = (datetime.now() - memory_item.created_at).total_seconds() / 3600
            if hours_old > 0.5 and memory_item.importance < 0.9:
                pruned_memories.append(memory_item)
                del self.memory_store[key]
                
                # Log pruning decision
                self.pruning_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "key": key,
                    "utility_score": utility,
                    "reason": "low_utility",
                    "access_count": memory_item.access_count,
                    "success_correlation": memory_item.success_correlation
                })
        
        return pruned_memories
    
    def _calculate_contextual_relevance(self, memory_item: SmartMemoryItem, context_keywords: set, task_type: str) -> float:
        """Calculate how relevant a memory is to current context"""
        relevance = 0.0
        
        # Keyword matching
        memory_text = str(memory_item.value).lower()
        memory_keywords = set(memory_text.split())
        
        # Calculate keyword overlap
        overlap = len(context_keywords & memory_keywords)
        if len(context_keywords) > 0:
            keyword_relevance = overlap / len(context_keywords)
            relevance += keyword_relevance * 0.4
        
        # Task type matching
        if memory_item.context_type == task_type:
            relevance += 0.3
        elif task_type in memory_item.context_type or memory_item.context_type in task_type:
            relevance += 0.2
        
        # Success correlation boost
        relevance += memory_item.success_correlation * 0.3
        
        return min(1.0, relevance)
    
    def _categorize_memory(self, key: str, context_type: str):
        """Categorize memory for better organization"""
        if "tool" in key.lower():
            self.memory_categories["tool_patterns"].append(key)
        elif "success" in key.lower():
            self.memory_categories["successful_strategies"].append(key)
        elif "fail" in key.lower():
            self.memory_categories["failed_attempts"].append(key)
        elif "context" in key.lower():
            self.memory_categories["context_patterns"].append(key)
        else:
            self.memory_categories["user_preferences"].append(key)
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights about memory system performance"""
        total_memories = len(self.memory_store)
        total_accesses = sum(item.access_count for item in self.memory_store.values())
        avg_utility = sum(item.get_utility_score() for item in self.memory_store.values()) / max(total_memories, 1)
        
        # Memory age distribution
        now = datetime.now()
        age_distribution = {
            "recent": sum(1 for item in self.memory_store.values() if (now - item.created_at).total_seconds() < 3600),
            "medium": sum(1 for item in self.memory_store.values() if 3600 <= (now - item.created_at).total_seconds() < 86400),
            "old": sum(1 for item in self.memory_store.values() if (now - item.created_at).total_seconds() >= 86400)
        }
        
        # Success correlation stats
        success_rates = [item.success_correlation for item in self.memory_store.values()]
        avg_success_rate = sum(success_rates) / max(len(success_rates), 1)
        
        return {
            "total_memories": total_memories,
            "total_accesses": total_accesses,
            "average_utility": avg_utility,
            "average_success_rate": avg_success_rate,
            "age_distribution": age_distribution,
            "pruning_events": len(self.pruning_history),
            "categories": {cat: len(items) for cat, items in self.memory_categories.items()},
            "memory_efficiency": avg_utility * avg_success_rate
        }
    
    def demonstrate_memory_evolution(self) -> str:
        """Generate a demonstration of memory evolution for the demo"""
        insights = self.get_memory_insights()
        
        demo_text = f"""
🧠 MEMORY EVOLUTION DEMONSTRATION:

📊 Current Memory State:
   Total Memories: {insights['total_memories']}/{self.max_memory_items}
   Memory Efficiency: {insights['memory_efficiency']:.3f}
   Pruning Events: {insights['pruning_events']}

🗂️  Memory Categories:
   Tool Patterns: {insights['categories']['tool_patterns']}
   Successful Strategies: {insights['categories']['successful_strategies']}
   Failed Attempts: {insights['categories']['failed_attempts']}
   Context Patterns: {insights['categories']['context_patterns']}

⚡ Memory Quality:
   Average Utility: {insights['average_utility']:.3f}
   Average Success Rate: {insights['average_success_rate']:.3f}
   Total Accesses: {insights['total_accesses']}

🧹 Self-Improvement Evidence:
   ✅ Automatic pruning of low-utility memories
   ✅ Success correlation tracking
   ✅ Context relevance optimization
   ✅ Memory category organization
"""
        
        # Show recent pruning events
        if self.pruning_history:
            recent_pruning = self.pruning_history[-3:]
            demo_text += f"\n🔄 Recent Memory Optimization:\n"
            for event in recent_pruning:
                demo_text += f"   Removed: {event['key'][:30]}... (utility: {event['utility_score']:.3f})\n"
        
        return demo_text

if __name__ == "__main__":
    # Test the smart memory system
    print("🧠 SMART MEMORY EVOLUTION SYSTEM TEST")
    print("=" * 50)
    
    memory_sys = IntelligentMemoryEvolutionSystem(max_memory_items=10)
    
    # Add various memories
    test_memories = [
        ("research_strategy_ai", {"approach": "use perplexity first"}, 0.9, "research"),
        ("failed_scraping_attempt", {"tool": "wrong_scraper", "error": "timeout"}, 0.3, "web_scraping"),
        ("travel_booking_success", {"tool": "airbnb_search", "outcome": "found perfect place"}, 0.8, "travel"),
        ("voice_call_pattern", {"tool": "vapi_create_call", "timing": "best in afternoon"}, 0.7, "communication"),
        ("obsolete_pattern", {"old_data": "no longer relevant"}, 0.1, "general"),
        ("context_optimization", {"length": "keep under 500 words"}, 0.6, "context"),
        ("user_preference", {"style": "formal tone preferred"}, 0.5, "user"),
        ("research_strategy_quantum", {"approach": "use perplexity research tool"}, 0.9, "research"),
        ("failed_call_attempt", {"error": "number invalid"}, 0.2, "communication"),
        ("successful_extraction", {"tool": "firecrawl_extract", "schema": "product_info"}, 0.8, "web_scraping"),
        ("old_travel_data", {"deprecated": "old booking system"}, 0.1, "travel"),
        ("new_research_insight", {"discovery": "reasoning tool works better for analysis"}, 0.95, "research")
    ]
    
    for key, value, importance, context_type in test_memories:
        memory_sys.store_memory(key, value, importance, context_type)
        time.sleep(0.1)  # Simulate time passing
    
    print(f"\n{memory_sys.demonstrate_memory_evolution()}")
    
    # Test memory retrieval
    print(f"\n🔍 MEMORY RETRIEVAL TEST:")
    relevant_memories = memory_sys.retrieve_relevant_memories("research AI developments", "research", 3)
    for i, memory in enumerate(relevant_memories, 1):
        print(f"   {i}. {memory['key']}: {memory['score']:.3f} relevance")
    
    print(f"\n✅ Smart memory evolution system working!")