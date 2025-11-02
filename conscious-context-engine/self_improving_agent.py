#!/usr/bin/env python3
"""
Self-Improving Agent Library
Installable package for adding self-improvement capabilities to any agent

Usage:
    from self_improving_agent import SelfImprovingAgentLibrary
    
    # Initialize with your MCP tools
    library = SelfImprovingAgentLibrary()
    library.register_tool("custom_tool", your_tool_instance)
    
    # Your agent automatically gets self-improvement capabilities
    result = await library.execute_with_improvement(user_request, task_type)
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pickle
import os
from mcp_client import (
    SelfImprovingMCPAgent, MCPTool, TaskContext, 
    ToolSelectionLearner, ContextOptimizer, MemoryEvolutionSystem
)

class SelfImprovingAgentLibrary:
    """
    Installable library that adds self-improvement capabilities to any agent.
    
    Key Features:
    - Tool Selection Learning: Learns which tools work best for different tasks
    - Context Optimization: Optimizes context windows for efficiency  
    - Memory Evolution: Builds and evolves knowledge from experience
    - Performance Tracking: Continuously improves based on outcomes
    
    Perfect for:
    - Research agents
    - Customer service bots
    - Content creation assistants
    - Data analysis tools
    - Any agent that uses multiple tools
    """
    
    def __init__(self, save_path: str = "./agent_knowledge", auto_save: bool = True):
        """
        Initialize the self-improving agent library
        
        Args:
            save_path: Directory to save learned knowledge
            auto_save: Whether to automatically save learned patterns
        """
        self.core_agent = SelfImprovingMCPAgent()
        self.save_path = save_path
        self.auto_save = auto_save
        self.custom_tools: Dict[str, MCPTool] = {}
        self.evaluation_function: Optional[Callable] = None
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Load existing knowledge if available
        self._load_knowledge()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def register_tool(self, name: str, tool: MCPTool):
        """Register a custom tool with the agent"""
        self.custom_tools[name] = tool
        self.core_agent.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")
    
    def set_evaluation_function(self, eval_func: Callable[[Dict[str, Any], str, str], float]):
        """
        Set custom evaluation function for outcome scoring
        
        Args:
            eval_func: Function that takes (result, user_intent, task_type) and returns 0-1 score
        """
        self.evaluation_function = eval_func
        self.core_agent._evaluate_outcome = eval_func
    
    async def execute_with_improvement(self, user_intent: str, task_type: str = "general") -> Dict[str, Any]:
        """
        Execute a task with automatic self-improvement
        
        Args:
            user_intent: What the user wants to accomplish
            task_type: Type of task (research, travel, communication, etc.)
            
        Returns:
            Dict containing result and improvement metrics
        """
        result = await self.core_agent.execute_task(user_intent, task_type)
        
        # Auto-save knowledge if enabled
        if self.auto_save:
            self._save_knowledge()
        
        return result
    
    def get_improvement_insights(self) -> Dict[str, Any]:
        """Get detailed insights about what the agent has learned"""
        insights = self.core_agent.get_learning_insights()
        
        # Add library-specific insights
        insights["custom_tools_registered"] = len(self.custom_tools)
        insights["knowledge_persistence"] = os.path.exists(os.path.join(self.save_path, "tool_learner.pkl"))
        insights["auto_save_enabled"] = self.auto_save
        
        return insights
    
    def reset_learning(self):
        """Reset all learned knowledge (useful for testing)"""
        self.core_agent.tool_learner = ToolSelectionLearner()
        self.core_agent.context_optimizer = ContextOptimizer()
        self.core_agent.memory_system = MemoryEvolutionSystem()
        self.core_agent.task_history = []
        self.core_agent.improvement_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "avg_execution_time": 0.0,
            "tool_efficiency": 0.0,
            "context_optimization": 0.0,
            "memory_utilization": 0.0
        }
        self.logger.info("Learning state reset")
    
    def export_knowledge(self, export_path: str):
        """Export learned knowledge to share with other agents"""
        export_data = {
            "tool_performance": self.core_agent.tool_learner.tool_performance,
            "success_combinations": self.core_agent.tool_learner.success_combinations,
            "context_effectiveness": self.core_agent.context_optimizer.context_effectiveness,
            "memory_store": dict(self.core_agent.memory_system.memory_store),
            "improvement_metrics": self.core_agent.improvement_metrics,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Knowledge exported to {export_path}")
    
    def import_knowledge(self, import_path: str):
        """Import knowledge from another agent"""
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        # Merge tool performance data
        for tool_key, performance in import_data.get("tool_performance", {}).items():
            if tool_key in self.core_agent.tool_learner.tool_performance:
                # Average the performance metrics
                existing = self.core_agent.tool_learner.tool_performance[tool_key]
                merged = {
                    "success_rate": (existing["success_rate"] + performance["success_rate"]) / 2,
                    "avg_execution_time": (existing["avg_execution_time"] + performance["avg_execution_time"]) / 2,
                    "avg_tokens": (existing["avg_tokens"] + performance["avg_tokens"]) / 2,
                    "usage_count": existing["usage_count"] + performance["usage_count"]
                }
                self.core_agent.tool_learner.tool_performance[tool_key] = merged
            else:
                self.core_agent.tool_learner.tool_performance[tool_key] = performance
        
        # Merge success combinations
        existing_combinations = self.core_agent.tool_learner.success_combinations
        imported_combinations = import_data.get("success_combinations", [])
        self.core_agent.tool_learner.success_combinations = existing_combinations + imported_combinations
        
        self.logger.info(f"Knowledge imported from {import_path}")
    
    def _save_knowledge(self):
        """Save learned knowledge to disk"""
        try:
            # Save tool learner
            with open(os.path.join(self.save_path, "tool_learner.pkl"), 'wb') as f:
                pickle.dump(self.core_agent.tool_learner, f)
            
            # Save context optimizer
            with open(os.path.join(self.save_path, "context_optimizer.pkl"), 'wb') as f:
                pickle.dump(self.core_agent.context_optimizer, f)
            
            # Save memory system
            with open(os.path.join(self.save_path, "memory_system.pkl"), 'wb') as f:
                pickle.dump(self.core_agent.memory_system, f)
            
            # Save metrics as JSON for readability
            with open(os.path.join(self.save_path, "metrics.json"), 'w') as f:
                json.dump(self.core_agent.improvement_metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
    
    def _load_knowledge(self):
        """Load previously saved knowledge"""
        try:
            # Load tool learner
            tool_learner_path = os.path.join(self.save_path, "tool_learner.pkl")
            if os.path.exists(tool_learner_path):
                with open(tool_learner_path, 'rb') as f:
                    self.core_agent.tool_learner = pickle.load(f)
            
            # Load context optimizer
            context_optimizer_path = os.path.join(self.save_path, "context_optimizer.pkl")
            if os.path.exists(context_optimizer_path):
                with open(context_optimizer_path, 'rb') as f:
                    self.core_agent.context_optimizer = pickle.load(f)
            
            # Load memory system
            memory_system_path = os.path.join(self.save_path, "memory_system.pkl")
            if os.path.exists(memory_system_path):
                with open(memory_system_path, 'rb') as f:
                    self.core_agent.memory_system = pickle.load(f)
            
            # Load metrics
            metrics_path = os.path.join(self.save_path, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.core_agent.improvement_metrics = json.load(f)
            
            self.logger.info("Previously learned knowledge loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge: {e}")

# Example custom tool implementation
class CustomAPIMCPTool(MCPTool):
    """Example of how to create a custom tool for the library"""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "custom_api"
    
    @property
    def capabilities(self) -> List[str]:
        return ["api_calls", "data_processing", "custom_logic"]
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Dict[str, Any]:
        # Implement your custom tool logic here
        await asyncio.sleep(0.1)  # Simulate API call
        
        return {
            "status": "success",
            "data": {
                "message": f"Custom API response for: {context.user_intent}",
                "parameters_used": parameters,
                "processing_time": 0.1
            },
            "tokens_used": 75
        }

# Usage examples and documentation
USAGE_EXAMPLES = """
# USAGE EXAMPLES

## 1. Basic Usage
```python
from self_improving_agent import SelfImprovingAgentLibrary

# Initialize the library
agent_lib = SelfImprovingAgentLibrary()

# Execute tasks - agent automatically improves
result1 = await agent_lib.execute_with_improvement("Research AI trends", "research")
result2 = await agent_lib.execute_with_improvement("Find hotels in NYC", "travel")

# Check what the agent learned
insights = agent_lib.get_improvement_insights()
print(f"Agent completed {insights['total_experience']} tasks")
print(f"Learned {insights['successful_patterns']} success patterns")
```

## 2. Adding Custom Tools
```python
from self_improving_agent import SelfImprovingAgentLibrary, CustomAPIMCPTool

# Create library instance
agent_lib = SelfImprovingAgentLibrary()

# Add your custom tool
custom_tool = CustomAPIMCPTool("https://api.example.com", "your-api-key")
agent_lib.register_tool("my_custom_tool", custom_tool)

# Agent will learn when to use your tool
result = await agent_lib.execute_with_improvement("Process data with custom API", "data_processing")
```

## 3. Custom Evaluation
```python
def my_evaluation_function(result, user_intent, task_type):
    # Your custom scoring logic
    if result.get("status") == "success":
        data_quality = len(result.get("data", {}))
        return min(data_quality / 100.0, 1.0)
    return 0.0

agent_lib.set_evaluation_function(my_evaluation_function)
```

## 4. Knowledge Sharing
```python
# Agent A learns from experience
agent_a = SelfImprovingAgentLibrary(save_path="./agent_a_knowledge")
await agent_a.execute_with_improvement("Research task", "research")

# Export A's knowledge
agent_a.export_knowledge("shared_knowledge.json")

# Agent B imports A's experience
agent_b = SelfImprovingAgentLibrary(save_path="./agent_b_knowledge") 
agent_b.import_knowledge("shared_knowledge.json")
# Agent B now has A's learned patterns
```

## 5. Enterprise Integration
```python
class EnterpriseAgent:
    def __init__(self):
        self.improvement_lib = SelfImprovingAgentLibrary(
            save_path="./enterprise_knowledge",
            auto_save=True
        )
        
        # Register enterprise tools
        self.improvement_lib.register_tool("salesforce", SalesforceTool())
        self.improvement_lib.register_tool("slack", SlackTool())
        self.improvement_lib.register_tool("jira", JiraTool())
    
    async def handle_user_request(self, request, request_type):
        # Automatically improves tool selection and context management
        return await self.improvement_lib.execute_with_improvement(request, request_type)
```

## Key Benefits:
✅ **Zero Configuration**: Works out of the box with any MCP tools
✅ **Automatic Learning**: Improves without manual tuning
✅ **Knowledge Persistence**: Learns across sessions  
✅ **Tool Agnostic**: Works with any tools you add
✅ **Performance Tracking**: Built-in metrics and insights
✅ **Knowledge Sharing**: Export/import learned patterns
✅ **Production Ready**: Handles errors, logging, persistence
"""

if __name__ == "__main__":
    print("🧠 Self-Improving Agent Library")
    print("=" * 40)
    print("Transform any agent into a self-improving system!")
    print(USAGE_EXAMPLES)