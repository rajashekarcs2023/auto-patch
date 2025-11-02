#!/usr/bin/env python3
"""
Simple Training Demo - Shows learning without ServerlessBackend complexity
Focus on demonstrating self-improvement for hackathon
"""
import asyncio
import time
from datetime import datetime
from mcp_client import SelfImprovingMCPAgent

class SimpleLearningDemo:
    """Simple demo showing clear learning progression"""
    
    def __init__(self):
        self.agent = SelfImprovingMCPAgent()
        self.training_log = []
    
    async def run_learning_session(self, num_episodes: int = 20):
        """Run learning session with clear progression tracking"""
        print("🧠 SELF-IMPROVING AGENT LEARNING SESSION")
        print("=" * 60)
        print(f"🎯 Training episodes: {num_episodes}")
        print(f"📊 Tools available: {len(self.agent.tools)}")
        print("👀 Watch tool selection improve over time!")
        print("=" * 60)
        
        # Training scenarios with expected optimal tools
        scenarios = [
            ("Research AI safety developments", "research", ["perplexity_search", "perplexity_research"]),
            ("Find luxury hotels in Tokyo", "travel", ["airbnb_search", "airbnb_listing_details"]),
            ("Create voice greeting message", "communication", ["vapi_synthesize", "vapi_create_call"]),
            ("Extract data from competitor website", "web_scraping", ["firecrawl_scrape", "firecrawl_extract"]),
            ("Get React documentation for hooks", "documentation", ["context7_resolve_library_id", "context7_get_library_docs"]),
            ("Research quantum computing trends", "research", ["perplexity_research", "perplexity_reason"]),
            ("Book accommodation in Paris", "travel", ["airbnb_search"]),
            ("Generate customer phone script", "communication", ["vapi_synthesize"]),
            ("Scrape product information", "web_scraping", ["firecrawl_extract", "real_firecrawl_extract"]),
            ("Find TypeScript 5.x documentation", "documentation", ["context7_get_library_docs"]),
        ]
        
        # Track learning metrics
        episode_metrics = []
        
        for episode in range(num_episodes):
            scenario_idx = episode % len(scenarios)
            task_desc, task_type, optimal_tools = scenarios[scenario_idx]
            
            print(f"\n🎯 EPISODE {episode + 1}/{num_episodes}")
            print(f"📝 Task: {task_desc}")
            print(f"🏷️  Type: {task_type}")
            
            # Get agent state before
            before_insights = self.agent.get_learning_insights()
            
            # Execute task
            start_time = time.time()
            result = await self.agent.execute_task(task_desc, task_type)
            execution_time = time.time() - start_time
            
            # Get agent state after
            after_insights = self.agent.get_learning_insights()
            
            # Evaluate tool choice
            tool_used = result['tool_used']
            is_optimal = any(tool_used.startswith(opt.split('_')[0]) for opt in optimal_tools)
            
            # Calculate learning score
            learning_score = 0.0
            if after_insights['total_experience'] > before_insights['total_experience']:
                learning_score += 0.3
            if after_insights['memory_items'] > before_insights['memory_items']:
                learning_score += 0.2
            if is_optimal:
                learning_score += 0.5
            
            # Store metrics
            episode_data = {
                "episode": episode + 1,
                "task_type": task_type,
                "tool_used": tool_used,
                "is_optimal": is_optimal,
                "success": result['success'],
                "execution_time": execution_time,
                "learning_score": learning_score,
                "memory_items": after_insights['memory_items'],
                "total_experience": after_insights['total_experience'],
                "memory_efficiency": after_insights.get('memory_efficiency', 0.0)
            }
            episode_metrics.append(episode_data)
            
            # Show results
            print(f"   🔧 Tool Used: {tool_used}")
            print(f"   🎯 Optimal Choice: {'✅ YES' if is_optimal else '❌ NO'}")
            if not is_optimal:
                print(f"   💡 Better Options: {', '.join(optimal_tools[:2])}")
            print(f"   ✅ Success: {result['success']}")
            print(f"   📊 Learning Score: {learning_score:.2f}")
            print(f"   💾 Memory Items: {after_insights['memory_items']}")
            print(f"   🎓 Experience: {after_insights['total_experience']}")
            
            # Show improvement if detected
            if episode > 0:
                prev_episode = episode_metrics[episode - 1]
                if episode_data['learning_score'] > prev_episode['learning_score']:
                    improvement = episode_data['learning_score'] - prev_episode['learning_score']
                    print(f"   📈 IMPROVEMENT: +{improvement:.2f} learning score!")
            
            # Brief pause for clarity
            await asyncio.sleep(0.5)
        
        # Show final analysis
        await self.show_learning_analysis(episode_metrics)
    
    async def show_learning_analysis(self, episode_metrics):
        """Show comprehensive learning analysis"""
        print(f"\n🎉 LEARNING SESSION COMPLETE!")
        print("=" * 60)
        
        # Calculate overall metrics
        total_episodes = len(episode_metrics)
        optimal_choices = sum(1 for ep in episode_metrics if ep['is_optimal'])
        avg_learning_score = sum(ep['learning_score'] for ep in episode_metrics) / total_episodes
        
        # Learning progression
        first_half = episode_metrics[:total_episodes//2]
        second_half = episode_metrics[total_episodes//2:]
        
        first_half_optimal = sum(1 for ep in first_half if ep['is_optimal']) / len(first_half)
        second_half_optimal = sum(1 for ep in second_half if ep['is_optimal']) / len(second_half)
        
        improvement = second_half_optimal - first_half_optimal
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Optimal Tool Choices: {optimal_choices}/{total_episodes} ({optimal_choices/total_episodes*100:.1f}%)")
        print(f"   Average Learning Score: {avg_learning_score:.3f}")
        
        print(f"\n📈 LEARNING PROGRESSION:")
        print(f"   First Half Optimal Rate: {first_half_optimal:.3f}")
        print(f"   Second Half Optimal Rate: {second_half_optimal:.3f}")
        print(f"   Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")
        
        # Memory and experience growth
        final_insights = self.agent.get_learning_insights()
        print(f"\n🧠 FINAL AGENT STATE:")
        print(f"   Total Experience: {final_insights['total_experience']} tasks")
        print(f"   Memory Items: {final_insights['memory_items']}")
        print(f"   Memory Efficiency: {final_insights.get('memory_efficiency', 0):.3f}")
        print(f"   Tool Patterns Learned: {len(final_insights['tool_performance'])}")
        
        # Task type specialization
        print(f"\n🎯 TASK TYPE SPECIALIZATION:")
        task_types = {}
        for ep in episode_metrics:
            task_type = ep['task_type']
            if task_type not in task_types:
                task_types[task_type] = {'total': 0, 'optimal': 0}
            task_types[task_type]['total'] += 1
            if ep['is_optimal']:
                task_types[task_type]['optimal'] += 1
        
        for task_type, stats in task_types.items():
            rate = stats['optimal'] / stats['total']
            print(f"   {task_type.title()}: {stats['optimal']}/{stats['total']} optimal ({rate:.2f})")
        
        # Show tool learning
        print(f"\n🔧 LEARNED TOOL PREFERENCES:")
        for tool_pattern, performance in final_insights['tool_performance'].items():
            if '_' in tool_pattern:
                tool_name, task_type = tool_pattern.split('_', 1)
                confidence = performance['success_rate']
                usage = performance['usage_count']
                print(f"   {tool_name} for {task_type}: {confidence:.3f} confidence ({usage} uses)")
        
        print(f"\n✨ LEARNING EVIDENCE:")
        print(f"   ✅ Tool selection improved by {improvement*100:+.1f}%")
        print(f"   ✅ Memory system built {final_insights['memory_items']} knowledge items")
        print(f"   ✅ Experience accumulated: {final_insights['total_experience']} tasks")
        print(f"   ✅ Specialization achieved across {len(task_types)} task types")
        
        # Save results
        import json
        results = {
            "training_completed": datetime.now().isoformat(),
            "episode_metrics": episode_metrics,
            "improvement_rate": improvement,
            "final_agent_state": final_insights,
            "specialization_by_task": task_types
        }
        
        with open("learning_session_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: learning_session_results.json")
        print(f"🎬 Ready for demo presentation!")

async def main():
    """Run the simple learning demo"""
    demo = SimpleLearningDemo()
    await demo.run_learning_session(num_episodes=15)

if __name__ == "__main__":
    asyncio.run(main())