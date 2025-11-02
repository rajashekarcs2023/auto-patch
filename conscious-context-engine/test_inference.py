#!/usr/bin/env python3
"""
Test script to diagnose why API inference calls are failing
"""
import asyncio
import traceback
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import art
from art.serverless.backend import ServerlessBackend

load_dotenv()

async def test_model_setup():
    """Test if model setup is working"""
    print("🔍 TESTING MODEL SETUP")
    print("=" * 50)
    
    # Check API key
    api_key = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_API_KEY exists: {bool(api_key)}")
    if api_key:
        print(f"WANDB_API_KEY starts with: {api_key[:10]}...")
    
    try:
        # Initialize model
        model = art.TrainableModel(
            name="test-model",
            project="test-project",
            base_model="OpenPipe/Qwen3-14B-Instruct"
        )
        
        backend = ServerlessBackend()
        await model.register(backend)
        
        print(f"✅ Model registered: {model.name}")
        print(f"✅ Inference URL: {model.inference_base_url}")
        print(f"✅ Inference name: {model.get_inference_name()}")
        print(f"✅ API key exists: {bool(model.inference_api_key)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        traceback.print_exc()
        return None

async def test_simple_inference(model):
    """Test a simple inference call"""
    print(f"\n🔍 TESTING SIMPLE INFERENCE")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    try:
        client = AsyncOpenAI(
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
        )
        
        print("✅ OpenAI client created")
        print(f"Base URL: {model.inference_base_url}")
        
        # Test simple message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ]
        
        print("🔄 Making inference call...")
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model.get_inference_name(),
                messages=messages,
                max_completion_tokens=50
            ),
            timeout=30.0
        )
        
        print("✅ Inference call successful!")
        print(f"Response: {response.choices[0].message.content}")
        
    except asyncio.TimeoutError:
        print("❌ Inference call timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Inference call failed: {type(e).__name__}: {e}")
        traceback.print_exc()

async def test_model_availability(model):
    """Test if the model is actually available for inference"""
    print(f"\n🔍 TESTING MODEL AVAILABILITY")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    try:
        # Check model step
        step = await model.get_step()
        print(f"✅ Current model step: {step}")
        
        # Try to get model info
        print(f"Model inference name: {model.get_inference_name()}")
        
        # Check if we need to wait for model to be ready
        if step == 0:
            print("⚠️  Model is at step 0 - might not be trained yet")
        else:
            print(f"✅ Model has been trained for {step} steps")
            
    except Exception as e:
        print(f"❌ Model availability check failed: {e}")
        traceback.print_exc()

async def test_multiple_calls(model, num_calls=5):
    """Test multiple inference calls to see failure rate"""
    print(f"\n🔍 TESTING MULTIPLE CALLS ({num_calls} calls)")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    successes = 0
    failures = 0
    
    for i in range(num_calls):
        try:
            print(f"Call {i+1}/{num_calls}... ", end="")
            
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model.get_inference_name(),
                    messages=[
                        {"role": "user", "content": f"Test call {i+1}"}
                    ],
                    max_completion_tokens=20
                ),
                timeout=15.0
            )
            
            print("✅ SUCCESS")
            successes += 1
            
        except asyncio.TimeoutError:
            print("❌ TIMEOUT")
            failures += 1
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}")
            failures += 1
    
    print(f"\n📊 RESULTS:")
    print(f"Successes: {successes}/{num_calls} ({successes/num_calls*100:.1f}%)")
    print(f"Failures: {failures}/{num_calls} ({failures/num_calls*100:.1f}%)")
    
    if failures > successes:
        print("⚠️  High failure rate detected - this explains the training issues!")

async def main():
    """Run all diagnostic tests"""
    print("🧪 DIAGNOSING API INFERENCE FAILURES")
    print("=" * 60)
    
    # Test model setup
    model = await test_model_setup()
    
    # Test model availability
    await test_model_availability(model)
    
    # Test simple inference
    await test_simple_inference(model)
    
    # Test multiple calls
    await test_multiple_calls(model, 10)
    
    print(f"\n🏁 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())