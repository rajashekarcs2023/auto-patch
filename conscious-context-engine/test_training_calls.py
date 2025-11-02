#!/usr/bin/env python3
"""
Test the exact same calls we make in training to find the issue
"""
import asyncio
import traceback
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import art
from art.serverless.backend import ServerlessBackend

load_dotenv()

async def test_training_model():
    """Test with the exact same model name as training"""
    print("🔍 TESTING WITH TRAINING MODEL NAME")
    print("=" * 50)
    
    try:
        # Use exact same model as training
        model = art.TrainableModel(
            name="agent-001",
            project="mcp-agent", 
            base_model="OpenPipe/Qwen3-14B-Instruct"
        )
        
        backend = ServerlessBackend()
        await model.register(backend)
        
        step = await model.get_step()
        print(f"✅ Training model registered: {model.name}")
        print(f"✅ Current step: {step}")
        print(f"✅ Inference name: {model.get_inference_name()}")
        
        return model
        
    except Exception as e:
        print(f"❌ Training model setup failed: {e}")
        traceback.print_exc()
        return None

async def test_training_message_format(model):
    """Test with exact same message format as training"""
    print(f"\n🔍 TESTING TRAINING MESSAGE FORMAT")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Exact same message format as training
    messages = [
        {
            "role": "system",
            "content": """You are an excellent MCP agent that learns optimal tool selection. Available tools: firecrawl_scrape, firecrawl_extract, perplexity_search, perplexity_research, airbnb_search, vapi_synthesize, context7_get_library_docs

For each task, choose the most appropriate tool and respond with: <tool_choice>tool_name</tool_choice>"""
        },
        {
            "role": "user", 
            "content": "Task: Research AI safety developments\nType: research\nChoose the best tool:"
        }
    ]
    
    try:
        print("🔄 Making training-style inference call...")
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
            ),
            timeout=30.0
        )
        
        print("✅ Training-style call successful!")
        content = response.choices[0].message.content
        print(f"Response: {content}")
        
        # Parse tool choice like in training
        chosen_tool = "unknown"
        if "<tool_choice>" in content and "</tool_choice>" in content:
            start = content.find("<tool_choice>") + len("<tool_choice>")
            end = content.find("</tool_choice>")
            chosen_tool = content[start:end].strip()
        
        print(f"Parsed tool: {chosen_tool}")
        
    except Exception as e:
        print(f"❌ Training-style call failed: {type(e).__name__}: {e}")
        traceback.print_exc()

async def test_concurrent_calls(model, num_concurrent=18):
    """Test concurrent calls like in training"""
    print(f"\n🔍 TESTING CONCURRENT CALLS ({num_concurrent} parallel)")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    async def single_call(call_id):
        """Single inference call"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an MCP agent. Choose a tool: <tool_choice>tool_name</tool_choice>"
                },
                {
                    "role": "user",
                    "content": f"Task {call_id}: Choose perplexity_search"
                }
            ]
            
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    max_completion_tokens=64,
                    messages=messages,
                    model=model.get_inference_name(),
                ),
                timeout=30.0
            )
            
            return {"id": call_id, "success": True, "content": response.choices[0].message.content}
            
        except Exception as e:
            return {"id": call_id, "success": False, "error": f"{type(e).__name__}: {e}"}
    
    print(f"🔄 Starting {num_concurrent} concurrent calls...")
    
    # Run all calls concurrently
    tasks = [single_call(i) for i in range(num_concurrent)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successes = 0
    failures = 0
    
    for result in results:
        if isinstance(result, dict):
            if result["success"]:
                successes += 1
                print(f"Call {result['id']}: ✅ SUCCESS")
            else:
                failures += 1
                print(f"Call {result['id']}: ❌ {result['error']}")
        else:
            failures += 1
            print(f"Call failed with exception: {result}")
    
    print(f"\n📊 CONCURRENT CALL RESULTS:")
    print(f"Successes: {successes}/{num_concurrent} ({successes/num_concurrent*100:.1f}%)")
    print(f"Failures: {failures}/{num_concurrent} ({failures/num_concurrent*100:.1f}%)")
    
    if failures > 0:
        print("⚠️  Found the issue! Concurrent calls are failing")
    else:
        print("✅ All concurrent calls succeeded")

async def test_art_retry_pattern(model):
    """Test the @art.retry pattern used in training"""
    print(f"\n🔍 TESTING ART.RETRY PATTERN")
    print("=" * 50)
    
    if not model:
        print("❌ No model available")
        return
    
    import requests
    
    @art.retry(exceptions=(requests.ReadTimeout,))
    async def test_rollout():
        client = AsyncOpenAI(
            base_url=model.inference_base_url,
            api_key=model.inference_api_key,
        )
        
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "Say 'test successful'"}
        ]
        
        response = await client.chat.completions.create(
            max_completion_tokens=32,
            messages=messages,
            model=model.get_inference_name(),
        )
        
        return response.choices[0].message.content
    
    try:
        print("🔄 Testing @art.retry pattern...")
        result = await test_rollout()
        print(f"✅ Art retry pattern works: {result}")
    except Exception as e:
        print(f"❌ Art retry pattern failed: {e}")
        traceback.print_exc()

async def main():
    """Run all training-specific diagnostic tests"""
    print("🧪 DIAGNOSING TRAINING-SPECIFIC API FAILURES")
    print("=" * 60)
    
    # Test with training model
    model = await test_training_model()
    
    # Test training message format
    await test_training_message_format(model)
    
    # Test concurrent calls (this is likely the issue)
    await test_concurrent_calls(model, 18)
    
    # Test art retry pattern
    await test_art_retry_pattern(model)
    
    print(f"\n🏁 TRAINING DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())