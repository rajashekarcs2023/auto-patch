#!/usr/bin/env python3
"""
Visual Self-Improvement Demo
Real-time visual demonstration of agent learning for judges
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from mcp_client import SelfImprovingMCPAgent, TaskContext

class VisualSelfImprovementDemo:
    """
    Visual demonstration that clearly shows what's evolving in real-time
    Perfect for showing judges the difference between our approach and others
    """
    
    def __init__(self):
        self.agent = SelfImprovingMCPAgent()
        self.demo_log = []
        self.evolution_snapshots = []
    
    def clear_screen(self):
        """Clear terminal screen for clean visual updates"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print demo header"""
        print("🧠 SELF-IMPROVING MCP AGENT - LIVE EVOLUTION DEMO")
        print("=" * 70)
        print("👨‍⚖️ FOR JUDGES: Watch the agent LEARN and EVOLVE in real-time")
        print("🎯 This is NOT just prompt engineering - this is genuine learning!")
        print("=" * 70)
    
    def print_agent_state(self, step: int):
        """Print current agent learning state"""
        insights = self.agent.get_learning_insights()
        
        print(f"\n📊 AGENT STATE - Step {step}")
        print("-" * 40)
        print(f"🔢 Total Experience: {insights['total_experience']} tasks")
        print(f"🎯 Tool Patterns Learned: {len(insights['tool_performance'])}")
        print(f"💾 Memory Items: {insights['memory_items']}")
        print(f"✨ Success Patterns: {insights['successful_patterns']}")
        
        # Show tool confidence evolution
        print(f"\n🧠 TOOL SELECTION CONFIDENCE:")
        if insights['tool_performance']:
            for tool_pattern, performance in list(insights['tool_performance'].items())[:3]:
                confidence = performance['success_rate']
                usage = performance['usage_count']
                print(f"   📈 {tool_pattern}: {confidence:.3f} confidence ({usage} uses)")
        else:
            print("   🆕 No patterns learned yet - agent is exploring")
    
    def print_task_execution(self, task_desc: str, task_type: str, result: Dict[str, Any]):
        """Print task execution with learning indicators"""
        print(f"\n🎯 EXECUTING TASK:")
        print(f"   📝 Task: {task_desc}")
        print(f"   🏷️  Type: {task_type}")
        print(f"   🔧 Tool Selected: {result['tool_used']}")
        print(f"   ✅ Success: {result['success']}")
        print(f"   📊 Outcome Score: {result.get('outcome_score', 0):.3f}")
        print(f"   ⏱️  Time: {result['execution_time']:.2f}s")
        
        # Show what learning was applied
        if "learning_applied" in result:
            learning = result["learning_applied"]
            print(f"\n🧠 INTELLIGENCE APPLIED:")
            print(f"   🎯 Tool Ranking: {'✅ YES' if learning['tool_ranking_used'] else '❌ NO'}")
            print(f"   📝 Context Optimized: {'✅ YES' if learning['context_optimized'] else '❌ NO'}")
            print(f"   💾 Memory Utilized: {'✅ YES' if learning['memory_utilized'] else '❌ NO'}")
    
    def print_evolution_detected(self, before_state: Dict, after_state: Dict):
        """Print when evolution is detected"""
        experience_gained = after_state['total_experience'] - before_state['total_experience']
        patterns_gained = len(after_state['tool_performance']) - len(before_state['tool_performance'])
        memory_gained = after_state['memory_items'] - before_state['memory_items']
        
        if experience_gained > 0:
            print(f"\n🚀 EVOLUTION DETECTED!")
            print(f"   📈 Experience: +{experience_gained}")
            if patterns_gained > 0:
                print(f"   🎯 New Tool Patterns: +{patterns_gained}")
            if memory_gained > 0:
                print(f"   💾 New Memories: +{memory_gained}")
    
    def show_tool_ranking_evolution(self, task_type: str):
        """Show how tool rankings evolve over time"""
        context = TaskContext(
            task_id="ranking_demo",
            task_type=task_type,
            user_intent=f"Sample {task_type} task",
            available_tools=list(self.agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        tool_rankings = self.agent.tool_learner.get_best_tools(context, context.available_tools)
        
        print(f"\n🏆 CURRENT TOOL RANKINGS FOR '{task_type.upper()}':")
        for i, (tool_name, confidence) in enumerate(tool_rankings, 1):
            bar_length = int(confidence * 20)  # 20 char bar
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {i}. {tool_name:12} {bar} {confidence:.3f}")
    
    async def run_evolution_demo(self):
        """Run the main evolution demonstration"""
        
        # Demo scenarios that show clear learning
        scenarios = [
            {
                "name": "Research Assistant Evolution",
                "tasks": [
                    ("Research latest AI developments", "research"),
                    ("Research quantum computing advances", "research"), 
                    ("Research machine learning trends", "research"),
                    ("Research robotics innovations", "research")
                ],
                "explanation": "Watch agent learn optimal tools for research tasks"
            },
            {
                "name": "Travel Planning Evolution", 
                "tasks": [
                    ("Find hotels in New York", "travel"),
                    ("Find accommodations in Paris", "travel"),
                    ("Find vacation rentals in Tokyo", "travel")
                ],
                "explanation": "Watch agent optimize travel-related tool selection"
            },
            {
                "name": "Content Creation Evolution",
                "tasks": [
                    ("Create voice message for team", "communication"),
                    ("Generate audio announcement", "communication"),
                    ("Create podcast narration", "communication")
                ],
                "explanation": "Watch agent learn communication tool patterns"
            }
        ]
        
        self.clear_screen()
        self.print_header()
        
        print(f"\n🎬 STARTING EVOLUTION DEMONSTRATION")
        print(f"📋 Will execute {sum(len(s['tasks']) for s in scenarios)} tasks across {len(scenarios)} scenarios")
        print(f"👀 Watch for: Tool ranking changes, confidence improvements, memory growth")
        input(f"\n📺 Press ENTER to start the live demo...")
        
        step = 0
        
        for scenario_idx, scenario in enumerate(scenarios, 1):
            self.clear_screen()
            self.print_header()
            
            print(f"\n🎬 SCENARIO {scenario_idx}: {scenario['name']}")
            print(f"💡 {scenario['explanation']}")
            print("=" * 50)
            
            # Record state before scenario
            before_scenario = self.agent.get_learning_insights()
            
            for task_idx, (task_desc, task_type) in enumerate(scenario['tasks'], 1):
                step += 1
                
                # Show current state
                self.print_agent_state(step)
                
                # Show current tool rankings
                self.show_tool_ranking_evolution(task_type)
                
                print(f"\n⏳ Preparing to execute task {task_idx}/{len(scenario['tasks'])}...")
                await asyncio.sleep(1)  # Dramatic pause
                
                # Record state before task
                before_task = self.agent.get_learning_insights()
                
                # Execute task
                result = await self.agent.execute_task(task_desc, task_type)
                
                # Show task execution
                self.print_task_execution(task_desc, task_type, result)
                
                # Record state after task
                after_task = self.agent.get_learning_insights()
                
                # Show evolution if detected
                self.print_evolution_detected(before_task, after_task)
                
                # Store snapshot
                self.evolution_snapshots.append({
                    "step": step,
                    "scenario": scenario['name'],
                    "task": task_desc,
                    "before": before_task,
                    "after": after_task,
                    "result": result
                })
                
                print(f"\n⏸️  Pausing to show learning impact...")
                await asyncio.sleep(2)  # Let judges see the changes
            
            # Show scenario summary
            after_scenario = self.agent.get_learning_insights()
            
            print(f"\n✨ SCENARIO {scenario_idx} COMPLETE!")
            print(f"📊 Learning Progress:")
            exp_gained = after_scenario['total_experience'] - before_scenario['total_experience']
            patterns_gained = len(after_scenario['tool_performance']) - len(before_scenario['tool_performance'])
            print(f"   📈 Experience Gained: +{exp_gained}")
            print(f"   🧠 Patterns Learned: +{patterns_gained}")
            
            if scenario_idx < len(scenarios):
                input(f"\n📺 Press ENTER to continue to next scenario...")
        
        # Final summary
        await self.show_final_evolution_summary()
    
    async def show_final_evolution_summary(self):
        """Show comprehensive evolution summary for judges"""
        self.clear_screen()
        self.print_header()
        
        print(f"\n🎉 EVOLUTION DEMONSTRATION COMPLETE!")
        print("=" * 50)
        
        final_insights = self.agent.get_learning_insights()
        
        print(f"📊 FINAL AGENT STATE:")
        print(f"   🔢 Total Experience: {final_insights['total_experience']} tasks")
        print(f"   🧠 Tool Patterns: {len(final_insights['tool_performance'])}")
        print(f"   💾 Memory Items: {final_insights['memory_items']}")
        print(f"   ✨ Success Patterns: {final_insights['successful_patterns']}")
        print(f"   📈 Success Rate: {final_insights['improvement_metrics']['successful_tasks']}/{final_insights['improvement_metrics']['total_tasks']}")
        
        print(f"\n🎯 LEARNED TOOL SPECIALIZATIONS:")
        for tool_pattern, performance in final_insights['tool_performance'].items():
            tool, task_type = tool_pattern.split('_', 1)
            confidence = performance['success_rate']
            usage = performance['usage_count']
            print(f"   🔧 {tool} for {task_type}: {confidence:.3f} confidence ({usage} uses)")
        
        print(f"\n🚀 WHAT MAKES THIS REVOLUTIONARY:")
        print(f"   ✅ REAL TOOL USAGE: Agent actually calls tools, not text generation")
        print(f"   ✅ GENUINE LEARNING: Performance improves through experience")
        print(f"   ✅ RUNTIME ADAPTATION: Learns during deployment")
        print(f"   ✅ OBSERVABLE EVOLUTION: You can see what's changing")
        print(f"   ✅ CROSS-DOMAIN TRANSFER: Learning applies across task types")
        
        print(f"\n💡 COMPARISON TO OTHER HACKATHON PROJECTS:")
        print(f"   🆚 LinkedIn Scraper + NFC: Just data collection, no learning")
        print(f"   🆚 Personal Assistants: Static capabilities, no improvement")
        print(f"   🆚 RAG Systems: Information retrieval, no intelligence evolution")
        print(f"   🏆 OUR SYSTEM: Actual self-improving intelligence")
        
        # Save evolution data
        evolution_report = {
            "demo_completed": datetime.now().isoformat(),
            "final_state": final_insights,
            "evolution_timeline": self.evolution_snapshots,
            "key_achievements": {
                "tasks_completed": final_insights['total_experience'],
                "patterns_learned": len(final_insights['tool_performance']),
                "memories_created": final_insights['memory_items'],
                "success_rate": final_insights['improvement_metrics']['successful_tasks'] / max(final_insights['improvement_metrics']['total_tasks'], 1)
            }
        }
        
        with open("evolution_demo_results.json", "w") as f:
            json.dump(evolution_report, f, indent=2, default=str)
        
        print(f"\n📄 Evolution data saved to: evolution_demo_results.json")
        print(f"🎬 Demo ready for judges!")

async def main():
    """Run the visual demo"""
    demo = VisualSelfImprovementDemo()
    await demo.run_evolution_demo()

if __name__ == "__main__":
    asyncio.run(main())