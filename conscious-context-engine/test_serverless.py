"""
Simple test to verify ServerlessBackend API connectivity
"""
import os
import asyncio
from dotenv import load_dotenv
import art
from art.serverless.backend import ServerlessBackend

load_dotenv()

async def test_serverless_backend():
    """Test basic ServerlessBackend connectivity"""
    
    print("🔍 Testing ServerlessBackend connectivity...")
    print(f"WANDB_API_KEY present: {'Yes' if os.getenv('WANDB_API_KEY') else 'No'}")
    
    if not os.getenv("WANDB_API_KEY"):
        print("❌ WANDB_API_KEY not found in environment")
        return False
    
    try:
        # Simple model for testing
        model = art.TrainableModel(
            name="test-connection-001",
            project="serverless-test", 
            base_model="OpenPipe/Qwen3-14B-Instruct",
        )
        
        print("🚀 Creating ServerlessBackend...")
        backend = ServerlessBackend()
        
        print("📡 Attempting to register model...")
        await model.register(backend)
        
        print("✅ ServerlessBackend connection successful!")
        print(f"✅ Model registered: {model.name}")
        print(f"✅ Inference URL: {model.inference_base_url}")
        
        # Test basic model info
        current_step = await model.get_step()
        print(f"✅ Current training step: {current_step}")
        
        return True
        
    except Exception as e:
        print(f"❌ ServerlessBackend connection failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        return False

async def main():
    """Run the serverless backend test"""
    print("=" * 60)
    print("SERVERLESS BACKEND CONNECTIVITY TEST")
    print("=" * 60)
    
    success = await test_serverless_backend()
    
    if success:
        print("\n🎉 ServerlessBackend is working correctly!")
    else:
        print("\n💥 ServerlessBackend test failed - check network/API key")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())