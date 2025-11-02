#!/usr/bin/env python3
"""
Self-Improving MCP Agent - Core Client Architecture
Revolutionary agent that learns to optimize tool usage, context, and memory
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from abc import ABC, abstractmethod

@dataclass
class ToolCall:
    """Represents a tool call with context and outcome"""
    tool_name: str
    parameters: Dict[str, Any]
    context: str
    timestamp: datetime
    success: bool
    response: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0

@dataclass
class TaskContext:
    """Represents the context for a task"""
    task_id: str
    task_type: str
    user_intent: str
    available_tools: List[str]
    context_window: List[str]
    memory_items: List[str]
    priority: int = 1

class MCPTool(ABC):
    """Abstract base class for MCP tools"""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        pass

class FirecrawlMCP(MCPTool):
    """Firecrawl MCP tool for web scraping and data extraction"""
    
    @property
    def name(self) -> str:
        return "firecrawl"
    
    @property
    def capabilities(self) -> List[str]:
        return ["web_scraping", "content_extraction", "pdf_processing", "data_structuring"]
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        # Mock implementation - replace with actual MCP client
        await asyncio.sleep(0.1)  # Simulate API call
        
        url = parameters.get("url", "")
        action = parameters.get("action", "scrape")
        
        if action == "scrape":
            return {
                "status": "success",
                "data": {
                    "url": url,
                    "title": f"Mock content from {url}",
                    "text": f"Extracted content for {context.user_intent}",
                    "metadata": {"word_count": 1500, "images": 3}
                },
                "tokens_used": 150
            }
        
        return {"status": "error", "error": f"Unknown action: {action}"}

class VapiMCP(MCPTool):
    """Vapi MCP tool for voice/audio processing"""
    
    @property
    def name(self) -> str:
        return "vapi"
    
    @property
    def capabilities(self) -> List[str]:
        return ["voice_synthesis", "speech_recognition", "audio_processing", "call_automation"]
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        
        action = parameters.get("action", "synthesize")
        text = parameters.get("text", "")
        
        if action == "synthesize":
            return {
                "status": "success",
                "data": {
                    "audio_url": f"https://mock-audio.com/{uuid.uuid4()}",
                    "duration": len(text) * 0.1,
                    "format": "mp3"
                },
                "tokens_used": 50
            }
        
        return {"status": "error", "error": f"Unknown action: {action}"}

class PerplexityMCP(MCPTool):
    """Perplexity MCP tool for research and information gathering"""
    
    @property
    def name(self) -> str:
        return "perplexity"
    
    @property
    def capabilities(self) -> List[str]:
        return ["research", "fact_checking", "current_events", "academic_search"]
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        await asyncio.sleep(0.3)
        
        query = parameters.get("query", "")
        search_type = parameters.get("type", "general")
        
        return {
            "status": "success",
            "data": {
                "query": query,
                "results": [
                    {
                        "title": f"Research result for: {query}",
                        "content": f"Detailed analysis relevant to {context.user_intent}",
                        "source": "mock-academic-source.com",
                        "confidence": 0.92
                    }
                ],
                "sources_count": 15,
                "search_type": search_type
            },
            "tokens_used": 200
        }

class AirbnbMCP(MCPTool):
    """Airbnb MCP tool for travel and accommodation services"""
    
    @property
    def name(self) -> str:
        return "airbnb"
    
    @property
    def capabilities(self) -> List[str]:
        return ["property_search", "booking_management", "travel_planning", "reviews_analysis"]
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        await asyncio.sleep(0.25)
        
        action = parameters.get("action", "search")
        location = parameters.get("location", "")
        
        if action == "search":
            return {
                "status": "success",
                "data": {
                    "location": location,
                    "properties": [
                        {
                            "id": f"prop_{uuid.uuid4().hex[:8]}",
                            "title": f"Beautiful home in {location}",
                            "price": 150,
                            "rating": 4.8,
                            "amenities": ["wifi", "kitchen", "parking"]
                        }
                    ],
                    "total_found": 247
                },
                "tokens_used": 100
            }
        
        return {"status": "error", "error": f"Unknown action: {action}"}

class ToolSelectionLearner:
    """Learns optimal tool selection patterns based on task outcomes"""
    
    def __init__(self):
        self.tool_performance: Dict[str, Dict[str, float]] = {}
        self.context_patterns: Dict[str, List[str]] = {}
        self.success_combinations: List[Dict[str, Any]] = []
    
    def record_tool_usage(self, tool_call: ToolCall, context: TaskContext, outcome_score: float):
        """Record tool usage and outcome for learning"""
        tool_key = f"{tool_call.tool_name}_{context.task_type}"
        
        if tool_key not in self.tool_performance:
            self.tool_performance[tool_key] = {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "avg_tokens": 0.0,
                "usage_count": 0
            }
        
        perf = self.tool_performance[tool_key]
        count = perf["usage_count"]
        
        # Update moving averages
        perf["success_rate"] = (perf["success_rate"] * count + outcome_score) / (count + 1)
        perf["avg_execution_time"] = (perf["avg_execution_time"] * count + tool_call.execution_time) / (count + 1)
        perf["avg_tokens"] = (perf["avg_tokens"] * count + tool_call.tokens_used) / (count + 1)
        perf["usage_count"] = count + 1
        
        # Record successful patterns
        if outcome_score > 0.8:
            self.success_combinations.append({
                "tool": tool_call.tool_name,
                "task_type": context.task_type,
                "context_length": len(context.context_window),
                "parameters": tool_call.parameters,
                "score": outcome_score
            })
    
    def get_best_tools(self, context: TaskContext, available_tools: List[str]) -> List[Tuple[str, float]]:
        """Get ranked list of best tools for given context"""
        tool_scores = []
        
        for tool_name in available_tools:
            tool_key = f"{tool_name}_{context.task_type}"
            
            if tool_key in self.tool_performance:
                perf = self.tool_performance[tool_key]
                # Composite score: success_rate weighted by usage confidence
                confidence = min(perf["usage_count"] / 10.0, 1.0)  # Max confidence at 10 uses
                score = perf["success_rate"] * confidence
                tool_scores.append((tool_name, score))
            else:
                # New tool gets medium score for exploration
                tool_scores.append((tool_name, 0.5))
        
        return sorted(tool_scores, key=lambda x: x[1], reverse=True)

class ContextOptimizer:
    """Learns to optimize context window management"""
    
    def __init__(self, max_context_length: int = 10):
        self.max_context_length = max_context_length
        self.context_effectiveness: Dict[str, float] = {}
        self.pruning_patterns: List[Dict[str, Any]] = []
    
    def optimize_context(self, context: TaskContext, task_history: List[str]) -> TaskContext:
        """Optimize context window based on learned patterns"""
        # Score each context item based on relevance to current task
        scored_context = []
        
        for item in context.context_window:
            relevance_score = self._calculate_relevance(item, context.user_intent, context.task_type)
            scored_context.append((item, relevance_score))
        
        # Sort by relevance and keep top items
        scored_context.sort(key=lambda x: x[1], reverse=True)
        optimized_context = [item for item, score in scored_context[:self.max_context_length]]
        
        # Add most recent history if space allows
        remaining_space = self.max_context_length - len(optimized_context)
        if remaining_space > 0 and task_history:
            optimized_context.extend(task_history[-remaining_space:])
        
        context.context_window = optimized_context
        return context
    
    def _calculate_relevance(self, context_item: str, user_intent: str, task_type: str) -> float:
        """Calculate relevance score for context item"""
        # Simple keyword matching - could be enhanced with embeddings
        intent_keywords = user_intent.lower().split()
        item_keywords = context_item.lower().split()
        
        overlap = len(set(intent_keywords) & set(item_keywords))
        return overlap / max(len(intent_keywords), 1)
    
    def record_context_outcome(self, context_used: List[str], outcome_score: float):
        """Record effectiveness of context configuration"""
        context_signature = f"len_{len(context_used)}_type_{hash(str(sorted(context_used[:3]))) % 1000}"
        
        if context_signature in self.context_effectiveness:
            current = self.context_effectiveness[context_signature]
            self.context_effectiveness[context_signature] = (current + outcome_score) / 2
        else:
            self.context_effectiveness[context_signature] = outcome_score

class MemoryEvolutionSystem:
    """Learns to evolve memory storage and retrieval patterns"""
    
    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.access_patterns: Dict[str, int] = {}
        self.memory_effectiveness: Dict[str, float] = {}
    
    def store_memory(self, key: str, value: Any, importance: float = 1.0):
        """Store memory with importance weighting"""
        self.memory_store[key] = {
            "value": value,
            "importance": importance,
            "created": datetime.now(),
            "access_count": 0,
            "last_accessed": datetime.now()
        }
    
    def retrieve_relevant_memories(self, context: TaskContext, max_memories: int = 5) -> List[str]:
        """Retrieve most relevant memories for current context"""
        # Score memories based on relevance and importance
        scored_memories = []
        
        for key, memory_data in self.memory_store.items():
            relevance = self._calculate_memory_relevance(key, memory_data["value"], context)
            importance = memory_data["importance"]
            recency = (datetime.now() - memory_data["last_accessed"]).total_seconds() / 3600  # Hours
            
            # Composite score: relevance * importance / log(recency + 1)
            score = relevance * importance / (1 + 0.1 * recency)
            scored_memories.append((key, score))
        
        # Return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [key for key, score in scored_memories[:max_memories]]
    
    def _calculate_memory_relevance(self, key: str, value: Any, context: TaskContext) -> float:
        """Calculate how relevant a memory is to current context"""
        # Simple implementation - could use embeddings for better matching
        key_words = set(key.lower().split())
        intent_words = set(context.user_intent.lower().split())
        
        overlap = len(key_words & intent_words)
        return overlap / max(len(key_words), 1)
    
    def evolve_memory_strategy(self, usage_patterns: Dict[str, int]):
        """Evolve memory storage and retrieval strategy based on usage"""
        # Identify frequently accessed patterns
        high_value_patterns = {k: v for k, v in usage_patterns.items() if v > 5}
        
        # Promote frequently used memory types
        for pattern in high_value_patterns:
            if pattern in self.memory_effectiveness:
                self.memory_effectiveness[pattern] *= 1.1  # Boost effectiveness
        
        # Prune low-value memories to make space
        current_time = datetime.now()
        to_remove = []
        
        for key, memory_data in self.memory_store.items():
            days_old = (current_time - memory_data["created"]).days
            if days_old > 7 and memory_data["access_count"] < 2:
                to_remove.append(key)
        
        for key in to_remove[:10]:  # Remove max 10 at a time
            del self.memory_store[key]

class SelfImprovingMCPAgent:
    """Main self-improving MCP agent that coordinates all components"""
    
    def __init__(self):
        # Import real tools
        from real_mcp_tools import get_all_real_mcp_tools
        from additional_mcp_tools import get_additional_mcp_tools
        
        # Combine all tools
        self.tools: Dict[str, MCPTool] = {}
        self.tools.update(get_all_real_mcp_tools())
        self.tools.update(get_additional_mcp_tools())
        
        self.tool_learner = ToolSelectionLearner()
        self.context_optimizer = ContextOptimizer()
        # Use intelligent memory system
        from smart_memory_system import IntelligentMemoryEvolutionSystem
        self.memory_system = IntelligentMemoryEvolutionSystem(max_memory_items=30)
        
        self.task_history: List[ToolCall] = []
        self.improvement_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "avg_execution_time": 0.0,
            "tool_efficiency": 0.0,
            "context_optimization": 0.0,
            "memory_utilization": 0.0
        }
    
    async def execute_task(self, user_intent: str, task_type: str = "general") -> Dict[str, Any]:
        """Execute a task with self-improving capabilities"""
        task_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create task context
        context = TaskContext(
            task_id=task_id,
            task_type=task_type,
            user_intent=user_intent,
            available_tools=list(self.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        # Optimize context using learned patterns with smart memory system
        relevant_memories = self.memory_system.retrieve_relevant_memories(user_intent, task_type, max_memories=5)
        context.memory_items = [mem["key"] for mem in relevant_memories]
        context = self.context_optimizer.optimize_context(context, [])
        
        # Get best tools for this task
        tool_rankings = self.tool_learner.get_best_tools(context, context.available_tools)
        
        # Execute with best tool
        best_tool_name = tool_rankings[0][0]
        tool = self.tools[best_tool_name]
        
        # Prepare parameters based on task intent
        parameters = self._prepare_tool_parameters(user_intent, task_type, best_tool_name)
        
        # Execute tool
        execution_start = datetime.now()
        try:
            result = await tool.execute(parameters, context)
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Create tool call record
            tool_call = ToolCall(
                tool_name=best_tool_name,
                parameters=parameters,
                context=user_intent,
                timestamp=execution_start,
                success=result.get("status") == "success",
                response=result,
                execution_time=execution_time,
                tokens_used=result.get("tokens_used", 0)
            )
            
            # Evaluate outcome
            outcome_score = self._evaluate_outcome(result, user_intent, task_type)
            
            # Learn from this execution
            self.tool_learner.record_tool_usage(tool_call, context, outcome_score)
            self.context_optimizer.record_context_outcome(context.context_window, outcome_score)
            
            # Store relevant information in intelligent memory system
            if outcome_score > 0.7:
                memory_key = f"{task_type}_{best_tool_name}_success_{len(self.task_history)}"
                self.memory_system.store_memory(
                    memory_key, 
                    {
                        "tool": best_tool_name,
                        "parameters": parameters,
                        "outcome_score": outcome_score,
                        "context": user_intent,
                        "execution_time": execution_time
                    }, 
                    importance=outcome_score,
                    context_type=task_type
                )
            elif outcome_score < 0.3:
                # Also store failures to learn from them
                memory_key = f"{task_type}_{best_tool_name}_failure_{len(self.task_history)}"
                self.memory_system.store_memory(
                    memory_key,
                    {
                        "tool": best_tool_name,
                        "parameters": parameters, 
                        "outcome_score": outcome_score,
                        "context": user_intent,
                        "lesson": "avoid_this_combination"
                    },
                    importance=0.3,
                    context_type=task_type
                )
            
            # Update metrics
            self._update_metrics(tool_call, outcome_score)
            
            # Add to history
            self.task_history.append(tool_call)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "task_id": task_id,
                "success": tool_call.success,
                "result": result,
                "execution_time": total_time,
                "tool_used": best_tool_name,
                "outcome_score": outcome_score,
                "improvement_metrics": self.improvement_metrics,
                "learning_applied": {
                    "tool_ranking_used": True,
                    "context_optimized": len(context.context_window) > 0,
                    "memory_utilized": len(context.memory_items) > 0
                }
            }
            
        except Exception as e:
            tool_call = ToolCall(
                tool_name=best_tool_name,
                parameters=parameters,
                context=user_intent,
                timestamp=execution_start,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - execution_start).total_seconds()
            )
            
            self.task_history.append(tool_call)
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "tool_used": best_tool_name
            }
    
    def _prepare_tool_parameters(self, user_intent: str, task_type: str, tool_name: str) -> Dict[str, Any]:
        """Prepare tool parameters based on intent and learned patterns"""
        # This could be enhanced with learned parameter patterns
        if tool_name == "firecrawl":
            if "scrape" in user_intent.lower() or "extract" in user_intent.lower():
                return {"action": "scrape", "url": "https://example.com"}
        elif tool_name == "vapi":
            if "voice" in user_intent.lower() or "audio" in user_intent.lower():
                return {"action": "synthesize", "text": user_intent}
        elif tool_name == "perplexity":
            return {"query": user_intent, "type": "research"}
        elif tool_name == "airbnb":
            if "travel" in user_intent.lower() or "accommodation" in user_intent.lower():
                return {"action": "search", "location": "San Francisco"}
        
        return {"query": user_intent}
    
    def _evaluate_outcome(self, result: Dict[str, Any], user_intent: str, task_type: str) -> float:
        """Evaluate how well the outcome matches the user intent"""
        if result.get("status") != "success":
            return 0.0
        
        # Simple evaluation - could be enhanced with LLM-based evaluation
        data = result.get("data", {})
        
        # Check if result contains expected elements
        score = 0.0
        
        if isinstance(data, dict):
            # Basic content check
            if any(key in data for key in ["text", "content", "results", "properties"]):
                score += 0.4
            
            # Relevance check (simple keyword matching)
            intent_keywords = set(user_intent.lower().split())
            result_text = str(data).lower()
            
            matching_keywords = sum(1 for keyword in intent_keywords if keyword in result_text)
            relevance_score = matching_keywords / max(len(intent_keywords), 1)
            score += relevance_score * 0.6
        
        return min(score, 1.0)
    
    def _update_metrics(self, tool_call: ToolCall, outcome_score: float):
        """Update agent performance metrics"""
        self.improvement_metrics["total_tasks"] += 1
        
        if tool_call.success:
            self.improvement_metrics["successful_tasks"] += 1
        
        # Update running averages
        total = self.improvement_metrics["total_tasks"]
        current_avg_time = self.improvement_metrics["avg_execution_time"]
        
        self.improvement_metrics["avg_execution_time"] = (
            (current_avg_time * (total - 1) + tool_call.execution_time) / total
        )
        
        # Tool efficiency (tokens per second)
        if tool_call.execution_time > 0:
            efficiency = tool_call.tokens_used / tool_call.execution_time
            current_efficiency = self.improvement_metrics["tool_efficiency"]
            self.improvement_metrics["tool_efficiency"] = (
                (current_efficiency * (total - 1) + efficiency) / total
            )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what the agent has learned"""
        memory_insights = self.memory_system.get_memory_insights()
        
        return {
            "tool_performance": self.tool_learner.tool_performance,
            "successful_patterns": len(self.tool_learner.success_combinations),
            "context_patterns": len(self.context_optimizer.context_effectiveness),
            "memory_items": memory_insights["total_memories"],
            "memory_efficiency": memory_insights["memory_efficiency"],
            "memory_pruning_events": memory_insights["pruning_events"],
            "memory_categories": memory_insights["categories"],
            "improvement_metrics": self.improvement_metrics,
            "total_experience": len(self.task_history)
        }

if __name__ == "__main__":
    # Example usage
    async def main():
        agent = SelfImprovingMCPAgent()
        
        # Test different types of tasks
        tasks = [
            ("Research the latest AI developments", "research"),
            ("Find accommodation in San Francisco", "travel"),
            ("Create a voice message for my team", "communication"),
            ("Extract data from a company website", "data_extraction")
        ]
        
        print("🧠 Self-Improving MCP Agent - Demo")
        print("=" * 50)
        
        for i, (intent, task_type) in enumerate(tasks, 1):
            print(f"\n🎯 Task {i}: {intent}")
            print(f"📋 Type: {task_type}")
            
            result = await agent.execute_task(intent, task_type)
            
            print(f"✅ Success: {result['success']}")
            print(f"🔧 Tool Used: {result['tool_used']}")
            print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
            print(f"📊 Outcome Score: {result.get('outcome_score', 0):.2f}")
            
            if "learning_applied" in result:
                learning = result["learning_applied"]
                print(f"🧠 Learning Applied:")
                print(f"   Tool Ranking: {learning['tool_ranking_used']}")
                print(f"   Context Optimized: {learning['context_optimized']}")
                print(f"   Memory Utilized: {learning['memory_utilized']}")
        
        print(f"\n📈 Learning Insights:")
        insights = agent.get_learning_insights()
        print(f"   Experience: {insights['total_experience']} tasks")
        print(f"   Success Rate: {insights['improvement_metrics']['successful_tasks']}/{insights['improvement_metrics']['total_tasks']}")
        print(f"   Memory Items: {insights['memory_items']}")
        print(f"   Learned Patterns: {insights['successful_patterns']}")
    
    asyncio.run(main())