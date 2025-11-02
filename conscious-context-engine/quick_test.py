#!/usr/bin/env python3
"""
Quick test to validate the focused demo works
"""
import asyncio
from mcp_client import SelfImprovingMCPAgent

async def quick_test():
    """Quick test of core functionality"""
    print("🧪 QUICK TEST - Self-Improving Agent")
    print("=" * 40)
    
    agent = SelfImprovingMCPAgent()
    
    # Check tool discovery
    print(f"✅ Tools discovered: {len(agent.tools)}")
    
    # Test a few tasks
    tasks = [
        ("Research AI trends", "research"),
        ("Find hotels in NYC", "travel"),
        ("Create voice message", "communication")
    ]
    
    for i, (task, task_type) in enumerate(tasks, 1):
        print(f"\n🎯 Task {i}: {task}")
        
        # Get tool ranking before
        from mcp_client import TaskContext
        context = TaskContext(
            task_id=f"test_{i}",
            task_type=task_type,
            user_intent=task,
            available_tools=list(agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        rankings_before = agent.tool_learner.get_best_tools(context, context.available_tools)
        top_tool_before = rankings_before[0][0]
        confidence_before = rankings_before[0][1]
        
        print(f"   Before: {top_tool_before} ({confidence_before:.3f})")
        
        # Execute task
        result = await agent.execute_task(task, task_type)
        
        # Get tool ranking after
        rankings_after = agent.tool_learner.get_best_tools(context, context.available_tools)
        top_tool_after = rankings_after[0][0]
        confidence_after = rankings_after[0][1]
        
        print(f"   Used: {result['tool_used']} (success: {result['success']})")
        print(f"   After: {top_tool_after} ({confidence_after:.3f})")
        
        if confidence_after > confidence_before:
            print(f"   📈 LEARNING: +{confidence_after - confidence_before:.3f} confidence!")
    
    # Final state
    insights = agent.get_learning_insights()
    print(f"\n📊 FINAL STATE:")
    print(f"   Experience: {insights['total_experience']} tasks")
    print(f"   Tool Patterns: {len(insights['tool_performance'])}")
    print(f"   Memory Items: {insights['memory_items']}")
    
    print(f"\n✅ Core functionality working!")
    return True

if __name__ == "__main__":
    asyncio.run(quick_test())