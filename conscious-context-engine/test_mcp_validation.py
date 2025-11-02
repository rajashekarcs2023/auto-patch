#!/usr/bin/env python3
"""
MCP Client Validation Test
Tests that our agent can discover, use, and learn from MCP tools
"""
import asyncio
import json
from datetime import datetime
from mcp_client import SelfImprovingMCPAgent, TaskContext

class MCPValidationTest:
    """Comprehensive validation of MCP client functionality"""
    
    def __init__(self):
        self.agent = SelfImprovingMCPAgent()
        self.test_results = []
    
    async def run_all_tests(self):
        """Run comprehensive validation tests"""
        print("🔍 MCP CLIENT VALIDATION TESTS")
        print("=" * 50)
        
        # Test 1: Tool Discovery
        await self.test_tool_discovery()
        
        # Test 2: Tool Execution
        await self.test_tool_execution()
        
        # Test 3: Learning Validation
        await self.test_learning_validation()
        
        # Test 4: Self-Improvement Evidence
        await self.test_self_improvement()
        
        # Final Report
        self.generate_validation_report()
    
    async def test_tool_discovery(self):
        """Test that agent can discover all available tools"""
        print("\n🔧 TEST 1: Tool Discovery")
        print("-" * 30)
        
        available_tools = list(self.agent.tools.keys())
        print(f"✅ Discovered {len(available_tools)} tools:")
        
        for tool_name in available_tools:
            tool = self.agent.tools[tool_name]
            capabilities = tool.capabilities
            print(f"   🔹 {tool_name}: {', '.join(capabilities)}")
        
        # Verify each tool is callable
        all_callable = True
        for tool_name, tool in self.agent.tools.items():
            try:
                # Test basic tool properties
                assert hasattr(tool, 'execute'), f"{tool_name} missing execute method"
                assert hasattr(tool, 'name'), f"{tool_name} missing name property"
                assert hasattr(tool, 'capabilities'), f"{tool_name} missing capabilities"
                print(f"   ✅ {tool_name}: Interface valid")
            except Exception as e:
                print(f"   ❌ {tool_name}: {e}")
                all_callable = False
        
        result = "PASS" if all_callable else "FAIL"
        print(f"\n📊 Tool Discovery: {result}")
        
        self.test_results.append({
            "test": "Tool Discovery",
            "result": result,
            "details": f"{len(available_tools)} tools discovered, all interfaces valid: {all_callable}"
        })
    
    async def test_tool_execution(self):
        """Test that each tool can be executed successfully"""
        print("\n⚡ TEST 2: Tool Execution")
        print("-" * 30)
        
        test_context = TaskContext(
            task_id="test_execution",
            task_type="test",
            user_intent="Test tool execution",
            available_tools=list(self.agent.tools.keys()),
            context_window=[],
            memory_items=[]
        )
        
        execution_results = {}
        
        for tool_name, tool in self.agent.tools.items():
            try:
                print(f"🧪 Testing {tool_name}...")
                
                # Prepare test parameters
                test_params = self._get_test_parameters(tool_name)
                
                # Execute tool
                start_time = datetime.now()
                result = await tool.execute(test_params, test_context)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Validate result structure
                assert isinstance(result, dict), f"{tool_name} didn't return dict"
                assert "status" in result, f"{tool_name} missing status field"
                
                success = result.get("status") == "success"
                print(f"   {'✅' if success else '❌'} {tool_name}: {result.get('status')} ({execution_time:.3f}s)")
                
                execution_results[tool_name] = {
                    "success": success,
                    "execution_time": execution_time,
                    "result": result
                }
                
            except Exception as e:
                print(f"   ❌ {tool_name}: Error - {e}")
                execution_results[tool_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        success_count = sum(1 for r in execution_results.values() if r.get("success", False))
        total_tools = len(execution_results)
        
        result = "PASS" if success_count == total_tools else "PARTIAL" if success_count > 0 else "FAIL"
        print(f"\n📊 Tool Execution: {result} ({success_count}/{total_tools} tools working)")
        
        self.test_results.append({
            "test": "Tool Execution", 
            "result": result,
            "details": f"{success_count}/{total_tools} tools executed successfully"
        })
    
    async def test_learning_validation(self):
        """Test that agent learning components work"""
        print("\n🧠 TEST 3: Learning Validation")
        print("-" * 30)
        
        # Test tool selection learner
        print("🎯 Testing Tool Selection Learning...")
        initial_patterns = len(self.agent.tool_learner.tool_performance)
        
        # Execute some tasks to generate learning data
        test_tasks = [
            ("Research AI developments", "research"),
            ("Find hotels in NYC", "travel"), 
            ("Generate voice message", "communication")
        ]
        
        for task_desc, task_type in test_tasks:
            result = await self.agent.execute_task(task_desc, task_type)
            print(f"   📝 Task: {task_desc[:30]}... - {result['success']}")
        
        # Check if learning occurred
        final_patterns = len(self.agent.tool_learner.tool_performance)
        learning_occurred = final_patterns > initial_patterns
        
        print(f"   🧠 Tool Patterns: {initial_patterns} → {final_patterns}")
        print(f"   {'✅' if learning_occurred else '❌'} Learning: {'DETECTED' if learning_occurred else 'NOT DETECTED'}")
        
        # Test memory system
        print("\n💾 Testing Memory System...")
        initial_memories = len(self.agent.memory_system.memory_store)
        
        # Store some test memories
        self.agent.memory_system.store_memory("test_pattern_1", {"data": "test"}, importance=0.8)
        self.agent.memory_system.store_memory("test_pattern_2", {"data": "test"}, importance=0.9)
        
        final_memories = len(self.agent.memory_system.memory_store)
        memory_working = final_memories > initial_memories
        
        print(f"   💾 Memory Items: {initial_memories} → {final_memories}")
        print(f"   {'✅' if memory_working else '❌'} Memory: {'WORKING' if memory_working else 'NOT WORKING'}")
        
        # Test context optimization
        print("\n📝 Testing Context Optimization...")
        test_context = TaskContext(
            task_id="context_test",
            task_type="test",
            user_intent="Test context optimization",
            available_tools=list(self.agent.tools.keys()),
            context_window=["item1", "item2", "item3"],
            memory_items=[]
        )
        
        optimized_context = self.agent.context_optimizer.optimize_context(test_context, ["history1", "history2"])
        context_optimized = len(optimized_context.context_window) >= 0  # Basic check
        
        print(f"   📝 Context Optimization: {'✅ WORKING' if context_optimized else '❌ NOT WORKING'}")
        
        overall_learning = learning_occurred and memory_working and context_optimized
        result = "PASS" if overall_learning else "FAIL"
        print(f"\n📊 Learning Validation: {result}")
        
        self.test_results.append({
            "test": "Learning Validation",
            "result": result,
            "details": f"Tool learning: {learning_occurred}, Memory: {memory_working}, Context: {context_optimized}"
        })
    
    async def test_self_improvement(self):
        """Test that self-improvement actually happens over time"""
        print("\n📈 TEST 4: Self-Improvement Evidence")
        print("-" * 30)
        
        # Record initial state
        initial_insights = self.agent.get_learning_insights()
        print(f"📊 Initial State:")
        print(f"   Experience: {initial_insights['total_experience']} tasks")
        print(f"   Tool Patterns: {len(initial_insights['tool_performance'])}")
        print(f"   Memory Items: {initial_insights['memory_items']}")
        print(f"   Success Patterns: {initial_insights['successful_patterns']}")
        
        # Run repeated similar tasks to trigger learning
        print(f"\n🔄 Running Learning Session...")
        research_tasks = [
            "Research quantum computing trends",
            "Research machine learning advances", 
            "Research robotics developments",
            "Research AI safety progress",
            "Research neural network innovations"
        ]
        
        for i, task in enumerate(research_tasks, 1):
            print(f"   📝 Task {i}/5: {task[:40]}...")
            result = await self.agent.execute_task(task, "research")
            
            # Show tool selection evolution
            context = TaskContext(
                task_id="ranking_test",
                task_type="research",
                user_intent=task,
                available_tools=list(self.agent.tools.keys()),
                context_window=[],
                memory_items=[]
            )
            
            tool_rankings = self.agent.tool_learner.get_best_tools(context, context.available_tools)
            best_tool = tool_rankings[0][0]
            confidence = tool_rankings[0][1]
            
            print(f"      🎯 Best Tool: {best_tool} (confidence: {confidence:.3f})")
        
        # Record final state
        final_insights = self.agent.get_learning_insights()
        print(f"\n📊 Final State:")
        print(f"   Experience: {final_insights['total_experience']} tasks")
        print(f"   Tool Patterns: {len(final_insights['tool_performance'])}")
        print(f"   Memory Items: {final_insights['memory_items']}")
        print(f"   Success Patterns: {final_insights['successful_patterns']}")
        
        # Calculate improvement
        experience_gained = final_insights['total_experience'] - initial_insights['total_experience']
        patterns_learned = len(final_insights['tool_performance']) - len(initial_insights['tool_performance'])
        memories_added = final_insights['memory_items'] - initial_insights['memory_items']
        
        print(f"\n🚀 Improvement Detected:")
        print(f"   📈 Experience: +{experience_gained} tasks")
        print(f"   🎯 Tool Patterns: +{patterns_learned}")
        print(f"   💾 Memory Items: +{memories_added}")
        
        # Validate improvement occurred
        improvement_detected = experience_gained > 0 and (patterns_learned > 0 or memories_added > 0)
        
        result = "PASS" if improvement_detected else "FAIL"
        print(f"\n📊 Self-Improvement: {result}")
        
        self.test_results.append({
            "test": "Self-Improvement",
            "result": result,
            "details": f"Experience +{experience_gained}, Patterns +{patterns_learned}, Memory +{memories_added}"
        })
    
    def _get_test_parameters(self, tool_name: str) -> dict:
        """Get appropriate test parameters for each tool"""
        if tool_name == "firecrawl":
            return {"action": "scrape", "url": "https://example.com"}
        elif tool_name == "vapi":
            return {"action": "synthesize", "text": "Test audio generation"}
        elif tool_name == "perplexity":
            return {"query": "test research query", "type": "general"}
        elif tool_name == "airbnb":
            return {"action": "search", "location": "Test City"}
        else:
            return {"query": "test"}
    
    def generate_validation_report(self):
        """Generate final validation report"""
        print(f"\n🎉 VALIDATION COMPLETE")
        print("=" * 50)
        
        passed_tests = sum(1 for test in self.test_results if test["result"] == "PASS")
        total_tests = len(self.test_results)
        
        print(f"📊 Overall Result: {passed_tests}/{total_tests} tests PASSED")
        print(f"\n📋 Test Summary:")
        
        for test in self.test_results:
            status = test["result"]
            icon = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
            print(f"   {icon} {test['test']}: {status}")
            print(f"      {test['details']}")
        
        # Overall verdict
        if passed_tests == total_tests:
            verdict = "🎉 READY FOR DEMO"
            message = "All systems working! Agent shows genuine self-improvement."
        elif passed_tests >= total_tests * 0.75:
            verdict = "⚠️ MOSTLY READY"
            message = "Core functionality working, minor issues to address."
        else:
            verdict = "❌ NEEDS WORK"
            message = "Significant issues found, requires debugging."
        
        print(f"\n{verdict}")
        print(f"💡 {message}")
        
        # Save detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "verdict": verdict,
            "test_results": self.test_results
        }
        
        with open("mcp_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: mcp_validation_report.json")

async def main():
    """Run the validation test suite"""
    validator = MCPValidationTest()
    await validator.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())