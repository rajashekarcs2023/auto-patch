"""
Test the specific WANDB API key provided
"""
import requests

def test_wandb_key(api_key):
    """Test specific W&B API key"""
    print(f"🔍 Testing W&B API key: {api_key[:8]}...")
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.wandb.ai/api/v1/viewer", headers=headers, timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ W&B API key is valid!")
            user_data = response.json()
            print(f"✅ User: {user_data.get('entity', 'Unknown')}")
            print(f"✅ Username: {user_data.get('username', 'Unknown')}")
            return True
        elif response.status_code == 401:
            print("❌ API key is invalid (401 Unauthorized)")
            return False
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    api_key = "918290b0eec0beae6b55fd9512ca9cab218b0834"
    print("=" * 60)
    print("TESTING SPECIFIC WANDB API KEY")
    print("=" * 60)
    
    success = test_wandb_key(api_key)
    
    if success:
        print("\n🎉 API key works! We can use ServerlessBackend")
    else:
        print("\n💥 API key doesn't work - need to check/update it")
    
    print("=" * 60)