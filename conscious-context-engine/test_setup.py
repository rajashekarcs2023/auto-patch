"""
Test script to verify our Conscious Context Engine setup
"""
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import art
        print("✅ ART imported successfully")
        
        from art.serverless.backend import ServerlessBackend
        print("✅ ServerlessBackend imported")
        
        import weave
        print("✅ Weave imported")
        
        from context_engine import ContextManager, ContextPolicy
        print("✅ Context engine imported")
        
        from research_env import ResearchEnvironment
        print("✅ Research environment imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_environment_variables():
    """Test that required environment variables are set"""
    print("\n🔑 Testing environment variables...")
    
    required_vars = {
        "WANDB_API_KEY": "ServerlessBackend",
        "FIRECRAWL_API_KEY": "Web research", 
        "OPENAI_API_KEY": "RULER evaluation"
    }
    
    all_good = True
    for var, purpose in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (for {purpose})")
        else:
            print(f"❌ {var}: Missing (needed for {purpose})")
            all_good = False
    
    return all_good

async def test_context_engine():
    """Test the context engine functionality"""
    print("\n🧠 Testing context engine...")
    
    try:
        from context_engine import ContextManager
        
        # Create context manager
        cm = ContextManager()
        
        # Add some test context
        chunk_id = cm.add_context(
            content="Test context about AI regulations",
            source="test"
        )
        print(f"✅ Added context chunk: {chunk_id}")
        
        # Test context selection
        selected, metrics = cm.get_optimized_context("regulation")
        print(f"✅ Context selection works: {len(selected)} chunks selected")
        print(f"   Metrics: {metrics}")
        
        # Test policy evolution
        evolution = cm.get_evolution_summary()
        print(f"✅ Evolution tracking: {evolution['evolution_stage']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Context engine error: {e}")
        return False

async def test_research_environment():
    """Test the research environment"""
    print("\n🔬 Testing research environment...")
    
    try:
        from research_env import ResearchEnvironment
        
        # Create environment (with empty API key for testing)
        env = ResearchEnvironment("")
        
        # Test task creation
        tasks = env.get_research_tasks()
        print(f"✅ Created {len(tasks)} research tasks")
        
        # Test scenario creation
        scenario = env.create_scenario(step=0)
        print(f"✅ Created scenario for task: {scenario.task.id}")
        
        # Test evaluation
        test_output = "Autonomous vehicles are regulated by NHTSA in the US"
        scores = env.evaluate_research_quality(test_output, scenario.task)
        print(f"✅ Evaluation works: {scores['overall']:.2f} overall score")
        
        return True
        
    except Exception as e:
        print(f"❌ Research environment error: {e}")
        return False

async def test_serverless_backend():
    """Test ServerlessBackend connection"""
    print("\n☁️ Testing ServerlessBackend connection...")
    
    try:
        from art.serverless.backend import ServerlessBackend
        import art
        
        # Test backend creation
        backend = ServerlessBackend()
        print("✅ ServerlessBackend created")
        
        # Test model creation (don't register yet, just create)
        model = art.TrainableModel(
            name="test-model",
            project="test-project",
            base_model="OpenPipe/Qwen3-14B-Instruct"
        )
        print("✅ TrainableModel created")
        
        print("⚠️ Skipping actual registration for test (would use GPU credits)")
        
        return True
        
    except Exception as e:
        print(f"❌ ServerlessBackend error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧠 Conscious Context Engine - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment Variables", test_environment_variables), 
        ("Context Engine", test_context_engine),
        ("Research Environment", test_research_environment),
        ("ServerlessBackend", test_serverless_backend)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        print("\n🚀 Next steps:")
        print("   1. Run: python main.py (to start training)")
        print("   2. Create demo UI for presentation")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())