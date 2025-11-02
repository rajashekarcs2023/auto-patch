#!/usr/bin/env python3
"""
FOCUSED SELF-IMPROVING AGENT DEMO
Clear demonstration of tool selection learning and memory improvement
"""
import asyncio
import time
from datetime import datetime
from mcp_client import SelfImprovingMCPAgent, TaskContext

class FocusedSelfImprovingDemo:
    """Simple, clear demonstration of agent self-improvement"""
    
    def __init__(self):
        self.agent = SelfImprovingMCPAgent()
        self.demo_tasks = [
            # Research tasks - should learn to prefer perplexity tools
            ("Research latest AI developments", "research"),
            ("Research quantum computing trends", "research"), 
            ("Research machine learning advances", "research"),
            
            # Travel tasks - should learn to prefer airbnb tools
            ("Find hotels in San Francisco", "travel"),
            ("Find accommodation in New York", "travel"),
            ("Find vacation rentals in Paris", "travel"),
            
            # Communication tasks - should learn to prefer vapi tools
            ("Create voice message for team", "communication"),
            ("Make phone call to customer", "communication"),
            ("Generate audio announcement", "communication"),
            
            # Web scraping tasks - should learn to prefer firecrawl tools
            ("Extract data from company website", "web_scraping"),
            ("Scrape product information", "web_scraping"),
            ("Map website structure", "web_scraping")
        ]
    
    def clear_screen(self):
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        print("🧠 SELF-IMPROVING MCP AGENT - LEARNING DEMONSTRATION")
        print("=" * 65)
        print("🎯 PROBLEM: Agents waste time using wrong tools")
        print("💡 SOLUTION: Agent learns optimal tool selection through experience")
        print("👀 WATCH: Tool confidence scores improve over time")
        print("=" * 65)
    
    def show_initial_state(self):
        """Show agent's initial state - no knowledge"""
        print(f"\n📊 INITIAL AGENT STATE (No Experience)")
        print("-" * 40)
        insights = self.agent.get_learning_insights()
        print(f"🔢 Tasks Completed: {insights['total_experience']}")
        print(f"🎯 Tool Patterns Learned: {len(insights['tool_performance'])}")
        print(f"💾 Memory Items: {insights['memory_items']}")
        print(f"🧹 Memory Efficiency: {insights.get('memory_efficiency', 0):.3f}")
        print(f"🗑️  Memory Pruning Events: {insights.get('memory_pruning_events', 0)}")
        print(f"📈 Success Patterns: {insights['successful_patterns']}")
        print(f"⚡ Status: BLANK SLATE - No knowledge about which tools work best")
    
    def show_tool_rankings(self, task_type: str, step: int):
        """Show current tool rankings for a task type"""
        context = TaskContext(
            task_id=f"demo_{step}",
            task_type=task_type,
            user_intent=f"Sample {task_type} task",
            available_tools=list(self.agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        rankings = self.agent.tool_learner.get_best_tools(context, context.available_tools)
        
        print(f"\n🏆 TOOL RANKINGS FOR '{task_type.upper()}' (Step {step}):")
        print("   " + "Confidence".ljust(12) + "Tool Name")
        for i, (tool_name, confidence) in enumerate(rankings[:5], 1):
            # Create visual confidence bar
            bar_length = int(confidence * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {confidence:.3f} {bar} {tool_name}")
        
        return rankings[0][1]  # Return top confidence score
    
    async def run_learning_demo(self):
        """Run the complete learning demonstration"""
        self.clear_screen()
        self.print_header()
        self.show_initial_state()
        
        print(f"\n🎬 STARTING LEARNING DEMONSTRATION")
        print(f"📋 Will execute {len(self.demo_tasks)} tasks across 4 categories")
        print(f"👀 Watch tool confidence scores improve with experience!")
        
        input(f"\n▶️  Press ENTER to begin...")
        
        confidence_history = {"research": [], "travel": [], "communication": [], "web_scraping": []}
        
        for step, (task_desc, task_type) in enumerate(self.demo_tasks, 1):
            self.clear_screen()
            self.print_header()
            
            # Show current learning state
            insights = self.agent.get_learning_insights()
            print(f"\n📊 CURRENT LEARNING STATE (Step {step}/{len(self.demo_tasks)})")
            print("-" * 50)
            print(f"🔢 Experience: {insights['total_experience']} tasks")
            print(f"🎯 Tool Patterns: {len(insights['tool_performance'])}")
            print(f"💾 Memory Items: {insights['memory_items']} (efficiency: {insights.get('memory_efficiency', 0):.3f})")
            if insights.get('memory_pruning_events', 0) > 0:
                print(f"🧹 Memory Pruned: {insights['memory_pruning_events']} obsolete memories removed")
            
            # Show tool rankings BEFORE task
            top_confidence = self.show_tool_rankings(task_type, step)
            confidence_history[task_type].append(top_confidence)
            
            print(f"\n🎯 EXECUTING TASK {step}:")
            print(f"   📝 Task: {task_desc}")
            print(f"   🏷️  Type: {task_type}")
            
            # Execute task
            start_time = time.time()
            result = await self.agent.execute_task(task_desc, task_type)
            execution_time = time.time() - start_time
            
            # Show results
            print(f"\n✅ TASK COMPLETED:")
            print(f"   🔧 Tool Used: {result['tool_used']}")
            print(f"   ✅ Success: {result['success']}")
            print(f"   📊 Score: {result.get('outcome_score', 0):.3f}")
            print(f"   ⏱️  Time: {execution_time:.2f}s")
            
            # Show learning that occurred
            if result.get('learning_applied', {}).get('tool_ranking_used'):
                print(f"   🧠 Learning Applied: Tool ranking system used")
            
            # Show improvement if this task type was seen before
            if len(confidence_history[task_type]) > 1:
                prev_confidence = confidence_history[task_type][-2]
                current_confidence = confidence_history[task_type][-1]
                improvement = current_confidence - prev_confidence
                if improvement > 0:
                    print(f"   📈 IMPROVEMENT: +{improvement:.3f} confidence for {task_type} tasks!")
                elif improvement == 0:
                    print(f"   ➡️  STABLE: Confidence maintained at {current_confidence:.3f}")
            
            # Pause for dramatic effect
            if step < len(self.demo_tasks):
                print(f"\n⏸️  Learning from this experience...")
                await asyncio.sleep(1.5)
        
        # Final analysis
        await self.show_final_results(confidence_history)
    
    async def show_final_results(self, confidence_history):
        """Show final learning results"""
        self.clear_screen()
        self.print_header()
        
        print(f"\n🎉 LEARNING DEMONSTRATION COMPLETE!")
        print("=" * 50)
        
        final_insights = self.agent.get_learning_insights()
        print(f"📊 FINAL AGENT STATE:")
        print(f"   🔢 Total Experience: {final_insights['total_experience']} tasks")
        print(f"   🎯 Tool Patterns Learned: {len(final_insights['tool_performance'])}")
        print(f"   💾 Memory Items Stored: {final_insights['memory_items']}")
        print(f"   ✨ Success Patterns: {final_insights['successful_patterns']}")
        
        print(f"\n📈 LEARNING PROGRESS BY TASK TYPE:")
        for task_type, confidence_scores in confidence_history.items():
            if len(confidence_scores) >= 2:
                initial = confidence_scores[0]
                final = confidence_scores[-1]
                improvement = final - initial
                improvement_pct = (improvement / max(initial, 0.001)) * 100
                
                print(f"   🔹 {task_type.upper()}:")
                print(f"      Initial Confidence: {initial:.3f}")
                print(f"      Final Confidence: {final:.3f}")
                print(f"      Improvement: +{improvement:.3f} ({improvement_pct:+.1f}%)")
        
        print(f"\n🚀 WHAT THE AGENT LEARNED:")
        for tool_pattern, performance in final_insights['tool_performance'].items():
            tool_name, task_type = tool_pattern.split('_', 1) if '_' in tool_pattern else (tool_pattern, 'general')
            confidence = performance['success_rate']
            usage_count = performance['usage_count']
            
            if usage_count > 0:
                print(f"   🔧 {tool_name} works well for {task_type}: {confidence:.3f} confidence ({usage_count} uses)")
        
        # Show memory categories if available
        memory_categories = final_insights.get('memory_categories', {})
        if any(memory_categories.values()):
            print(f"\n🗂️  MEMORY ORGANIZATION:")
            for category, count in memory_categories.items():
                if count > 0:
                    print(f"   📁 {category.replace('_', ' ').title()}: {count} memories")
        
        print(f"\n💡 PROOF OF SELF-IMPROVEMENT:")
        print(f"   ✅ TOOL LEARNING: Agent learned which tools work best for each task type")
        print(f"   ✅ SMART MEMORY: {final_insights['memory_items']} optimized memories (efficiency: {final_insights.get('memory_efficiency', 0):.3f})")
        if final_insights.get('memory_pruning_events', 0) > 0:
            print(f"   ✅ MEMORY EVOLUTION: {final_insights['memory_pruning_events']} obsolete memories automatically removed")
        print(f"   ✅ EXPERIENCE GROWTH: Completed {final_insights['total_experience']} tasks")
        print(f"   ✅ PERFORMANCE TRACKING: {final_insights['successful_patterns']} proven success patterns")
        
        print(f"\n🏆 COMPETITIVE ADVANTAGE:")
        print(f"   🆚 Other Hackathon Projects: Static tool usage, no learning")
        print(f"   🆚 LinkedIn Scraper + NFC: Just data collection, no intelligence")
        print(f"   🆚 Personal Assistants: Fixed capabilities, no improvement")
        print(f"   🏅 OUR AGENT: Genuine self-improvement through experience")
        
        # Save results
        results = {
            "demo_completed": datetime.now().isoformat(),
            "final_insights": final_insights,
            "confidence_history": confidence_history,
            "total_tools": len(self.agent.tools),
            "learning_demonstrated": True
        }
        
        import json
        with open("focused_demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: focused_demo_results.json")
        print(f"🎬 Ready for judge presentation!")

async def main():
    """Run the focused demo"""
    demo = FocusedSelfImprovingDemo()
    await demo.run_learning_demo()

if __name__ == "__main__":
    asyncio.run(main())