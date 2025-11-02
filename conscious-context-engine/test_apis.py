"""
Test all our API connections
"""
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import requests

load_dotenv()

async def test_openai_api():
    """Test OpenAI API connectivity"""
    print("🔍 Testing OpenAI API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, test message"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API working!")
        print(f"✅ Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

def test_wandb_api():
    """Test W&B API connectivity"""
    print("🔍 Testing W&B API...")
    
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("❌ WANDB_API_KEY not found")
        return False
    
    try:
        # Simple API call to check connectivity
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.wandb.ai/api/v1/viewer", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ W&B API working!")
            user_data = response.json()
            print(f"✅ User: {user_data.get('entity', 'Unknown')}")
            return True
        else:
            print(f"❌ W&B API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ W&B API failed: {e}")
        return False

async def test_all_apis():
    """Test all API connections"""
    print("=" * 60)
    print("API CONNECTIVITY TEST")
    print("=" * 60)
    
    results = {}
    
    # Test APIs
    results['openai'] = await test_openai_api()
    print()
    results['wandb'] = test_wandb_api()
    print()
    
    # Summary
    print("📊 RESULTS SUMMARY:")
    for api, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {api.upper()}: {'Working' if status else 'Failed'}")
    
    all_working = all(results.values())
    if all_working:
        print("\n🎉 All APIs are working! ServerlessBackend issue might be temporary.")
        print("💡 Recommendation: Try ServerlessBackend again or use LocalBackend as fallback")
    else:
        print("\n⚠️  Some APIs are failing. Check your API keys and network connection.")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_all_apis())