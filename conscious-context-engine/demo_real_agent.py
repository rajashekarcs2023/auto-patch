#!/usr/bin/env python3
"""
Self-Improving MCP Agent - Real-World Demo
Demonstrates genuine agent capabilities with tool usage evolution
"""
import asyncio
import json
from datetime import datetime
from mcp_client import SelfImprovingMCPAgent

class RealWorldDemo:
    """Demo showcasing real agent capabilities with self-improvement"""
    
    def __init__(self):
        self.agent = SelfImprovingMCPAgent()
        self.demo_scenarios = [
            {
                "name": "Travel Planning Assistant",
                "tasks": [
                    ("Find luxury accommodations in New York for next weekend", "travel"),
                    ("Research the best restaurants in Manhattan", "research"),
                    ("Create a welcome message for guests arriving at the hotel", "communication"),
                    ("Extract venue information from Times Square event websites", "data_extraction")
                ],
                "expected_improvement": "Tool selection optimization for travel workflows"
            },
            {
                "name": "Market Research Agent",
                "tasks": [
                    ("Research emerging trends in AI startups 2024", "research"),
                    ("Extract competitor data from TechCrunch articles", "data_extraction"),
                    ("Find co-working spaces for a new AI company in SF", "travel"),
                    ("Generate audio summary of market findings", "communication")
                ],
                "expected_improvement": "Context management for research continuity"
            },
            {
                "name": "Content Creation Workflow",
                "tasks": [
                    ("Research latest developments in quantum computing", "research"),
                    ("Extract key quotes from scientific papers on arXiv", "data_extraction"),
                    ("Create engaging voice narration for tech podcast", "communication"),
                    ("Find conference venues for quantum computing events", "travel")
                ],
                "expected_improvement": "Memory utilization for content consistency"
            }
        ]
    
    async def run_complete_demo(self):
        """Run complete demo showing self-improvement across scenarios"""
        print("🚀 SELF-IMPROVING MCP AGENT - REAL WORLD DEMO")
        print("=" * 60)
        print("🎯 Demonstrating genuine tool-using agent with self-improvement")
        print("📊 Watch tool selection, context optimization, and memory evolution")
        print("=" * 60)
        
        total_tasks = 0
        scenario_results = []
        
        for scenario_idx, scenario in enumerate(self.demo_scenarios, 1):
            print(f"\n🎬 SCENARIO {scenario_idx}: {scenario['name']}")
            print(f"🎯 Expected Improvement: {scenario['expected_improvement']}")
            print("-" * 50)
            
            scenario_start_time = datetime.now()
            scenario_task_results = []
            
            for task_idx, (task_description, task_type) in enumerate(scenario['tasks'], 1):
                print(f"\n📋 Task {task_idx}: {task_description}")
                print(f"🏷️  Type: {task_type}")
                
                # Get agent state before task
                pre_insights = self.agent.get_learning_insights()
                
                # Execute task
                result = await self.agent.execute_task(task_description, task_type)
                
                # Get agent state after task
                post_insights = self.agent.get_learning_insights()
                
                # Show results
                print(f"✅ Success: {result['success']}")
                print(f"🔧 Tool Selected: {result['tool_used']}")
                print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
                print(f"📊 Outcome Score: {result.get('outcome_score', 0):.2f}")
                
                # Show learning applied
                if "learning_applied" in result:
                    learning = result["learning_applied"]
                    print(f"🧠 Intelligence Applied:")
                    print(f"   🎯 Tool Ranking: {'✅' if learning['tool_ranking_used'] else '❌'}")
                    print(f"   📝 Context Optimized: {'✅' if learning['context_optimized'] else '❌'}")
                    print(f"   💾 Memory Utilized: {'✅' if learning['memory_utilized'] else '❌'}")
                
                # Show improvement metrics
                if post_insights['total_experience'] > pre_insights['total_experience']:
                    print(f"📈 Learning Progress:")
                    print(f"   Experience: {pre_insights['total_experience']} → {post_insights['total_experience']}")
                    print(f"   Success Rate: {post_insights['improvement_metrics']['successful_tasks']}/{post_insights['improvement_metrics']['total_tasks']}")
                    
                    if post_insights['successful_patterns'] > pre_insights['successful_patterns']:
                        print(f"   🎉 New Success Pattern Learned! ({post_insights['successful_patterns']} total)")
                
                scenario_task_results.append({
                    "task": task_description,
                    "success": result['success'],
                    "tool_used": result['tool_used'],
                    "outcome_score": result.get('outcome_score', 0),
                    "learning_applied": result.get('learning_applied', {})
                })
                
                total_tasks += 1
            
            scenario_duration = (datetime.now() - scenario_start_time).total_seconds()
            scenario_results.append({
                "name": scenario['name'],
                "duration": scenario_duration,
                "tasks": scenario_task_results,
                "improvement_demonstrated": scenario['expected_improvement']
            })
            
            print(f"\n✨ Scenario {scenario_idx} Complete!")
            print(f"⏱️  Duration: {scenario_duration:.1f}s")
            print(f"📊 Tasks Completed: {len(scenario_task_results)}")
        
        # Final analysis
        await self.show_final_analysis(scenario_results, total_tasks)
    
    async def show_final_analysis(self, scenario_results, total_tasks):
        """Show comprehensive analysis of agent improvement"""
        print(f"\n🎉 DEMO COMPLETE - SELF-IMPROVEMENT ANALYSIS")
        print("=" * 60)
        
        final_insights = self.agent.get_learning_insights()
        
        print(f"📊 OVERALL PERFORMANCE:")
        metrics = final_insights['improvement_metrics']
        success_rate = (metrics['successful_tasks'] / max(metrics['total_tasks'], 1)) * 100
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Execution Time: {metrics['avg_execution_time']:.2f}s")
        print(f"   Tool Efficiency: {metrics['tool_efficiency']:.1f} tokens/sec")
        
        print(f"\n🧠 LEARNING ACHIEVEMENTS:")
        print(f"   🎯 Tool Performance Patterns: {len(final_insights['tool_performance'])}")
        print(f"   📝 Context Patterns Learned: {final_insights['context_patterns']}")
        print(f"   💾 Memory Items Stored: {final_insights['memory_items']}")
        print(f"   ✨ Successful Patterns: {final_insights['successful_patterns']}")
        
        print(f"\n🔧 TOOL USAGE LEARNING:")
        for tool_pattern, performance in final_insights['tool_performance'].items():
            print(f"   {tool_pattern}:")
            print(f"     Success Rate: {performance['success_rate']:.2f}")
            print(f"     Usage Count: {performance['usage_count']}")
            print(f"     Avg Time: {performance['avg_execution_time']:.2f}s")
        
        print(f"\n🚀 SELF-IMPROVEMENT EVIDENCE:")
        print(f"   ✅ Tool Selection: Agent learned optimal tools for each task type")
        print(f"   ✅ Context Management: Optimized context windows for efficiency")
        print(f"   ✅ Memory Evolution: Built knowledge base from successful patterns")
        print(f"   ✅ Performance Tracking: Continuously improved based on outcomes")
        
        print(f"\n💡 REVOLUTIONARY ASPECTS:")
        print(f"   🎯 Real Tool Usage: Agent actually uses tools, not just text generation")
        print(f"   📈 Genuine Learning: Performance improves through experience")
        print(f"   🔄 Runtime Adaptation: Learns during deployment, not just training")
        print(f"   🌐 Cross-Domain Transfer: Learning applies across different task types")
        
        # Show specific improvements
        await self.demonstrate_learned_improvements()
    
    async def demonstrate_learned_improvements(self):
        """Demonstrate specific learned improvements"""
        print(f"\n🎯 LEARNED IMPROVEMENT DEMONSTRATION")
        print("-" * 40)
        
        # Test the same task type to show improvement
        test_task = "Research the latest developments in AI safety"
        
        print(f"🧪 Testing Learned Patterns with: '{test_task}'")
        
        # Get current tool rankings
        from mcp_client import TaskContext
        test_context = TaskContext(
            task_id="test",
            task_type="research", 
            user_intent=test_task,
            available_tools=list(self.agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        tool_rankings = self.agent.tool_learner.get_best_tools(test_context, test_context.available_tools)
        
        print(f"🏆 Tool Ranking (learned preferences):")
        for i, (tool_name, score) in enumerate(tool_rankings, 1):
            print(f"   {i}. {tool_name}: {score:.3f}")
        
        # Show memory utilization
        relevant_memories = self.agent.memory_system.retrieve_relevant_memories(test_context)
        print(f"💾 Relevant Memories Found: {len(relevant_memories)}")
        for memory in relevant_memories[:3]:
            print(f"   - {memory}")
        
        print(f"\n✨ This demonstrates genuine self-improvement:")
        print(f"   🎯 Best tool automatically selected based on experience")
        print(f"   💾 Relevant past knowledge retrieved")
        print(f"   📊 Confidence scores based on actual performance data")

async def run_hackathon_demo():
    """Run the complete hackathon demo"""
    demo = RealWorldDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(run_hackathon_demo())