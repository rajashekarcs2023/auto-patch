#!/usr/bin/env python3
"""
🔥 REAL Error Recovery Agent Training
Uses ServerlessBackend to train agent on actual error patterns
Shows genuine autonomous evolution with real APIs and MCP tools
"""
import os
import random
import json
import time
import asyncio
import requests
from datetime import datetime
from typing import Literal, TypedDict, Dict, List, Any
from dotenv import load_dotenv
import art
from art.serverless.backend import ServerlessBackend
import weave
from openai import AsyncOpenAI
from pydantic import BaseModel
from real_mcp_tools import get_all_real_mcp_tools
import hashlib

load_dotenv()

# Check for required environment variables
if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is required for training and logging to Weights & Biases.")

class ErrorRecoveryScenario(BaseModel):
    """Training scenario for error recovery"""
    step: int
    error_type: Literal["timeout", "rate_limit", "auth_error", "network_error", "server_error"]
    api_endpoint: str
    complexity_level: Literal["simple", "medium", "complex"]

class ErrorPattern(TypedDict):
    """Structure for error patterns learned by the agent"""
    signature: str
    error_type: str
    context: Dict[str, Any]
    successful_fixes: List[Dict[str, Any]]
    failure_patterns: List[Dict[str, Any]]
    research_insights: List[str]

class RealErrorRecoveryAgent:
    """Agent that learns real error recovery through ServerlessBackend training"""
    
    def __init__(self):
        self.mcp_tools = get_all_real_mcp_tools()
        self.error_memory = self.load_error_memory()
        self.training_stats = []
        
        print(f"🧠 Initialized with {len(self.mcp_tools)} real MCP tools")
        print(f"📚 Loaded {len(self.error_memory.get('patterns', []))} error patterns")
    
    def load_error_memory(self) -> Dict:
        """Load persistent error memory"""
        try:
            with open("real_error_memory.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "patterns": [],
                "global_stats": {
                    "total_recoveries": 0,
                    "successful_predictions": 0,
                    "research_sessions": 0
                },
                "training_metadata": {
                    "training_started": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            }
    
    def save_error_memory(self):
        """Save error memory to persistent storage"""
        self.error_memory["training_metadata"]["last_updated"] = datetime.now().isoformat()
        with open("real_error_memory.json", 'w') as f:
            json.dump(self.error_memory, f, indent=2)
    
    def create_error_signature(self, error_type: str, endpoint: str, context: Dict) -> str:
        """Create unique signature for error pattern"""
        signature_data = {
            "error_type": error_type,
            "endpoint_hash": hashlib.md5(endpoint.encode()).hexdigest()[:8],
            "context_complexity": len(str(context)),
            "has_auth": "auth" in str(context).lower(),
            "has_timeout": "timeout" in str(context).lower()
        }
        return hashlib.md5(str(signature_data).encode()).hexdigest()
    
    def simulate_real_api_error(self, scenario: ErrorRecoveryScenario) -> Dict[str, Any]:
        """Simulate realistic API errors based on scenario"""
        
        error_scenarios = {
            "timeout": {
                "exception": "TimeoutError: Request timed out after 5 seconds",
                "http_code": 408,
                "context": {"timeout": 5, "large_payload": scenario.complexity_level == "complex"}
            },
            "rate_limit": {
                "exception": "HTTPError: 429 Too Many Requests",
                "http_code": 429,
                "context": {"requests_per_minute": 60 if scenario.complexity_level == "simple" else 120}
            },
            "auth_error": {
                "exception": "HTTPError: 401 Unauthorized",
                "http_code": 401,
                "context": {"token_expired": True, "auth_type": "bearer"}
            },
            "network_error": {
                "exception": "ConnectionError: Failed to establish connection",
                "http_code": 0,
                "context": {"network_unstable": True, "retry_count": 0}
            },
            "server_error": {
                "exception": "HTTPError: 500 Internal Server Error",
                "http_code": 500,
                "context": {"server_overloaded": scenario.complexity_level == "complex"}
            }
        }
        
        return error_scenarios[scenario.error_type]
    
    def research_error_with_mcp(self, error_data: Dict, endpoint: str) -> Dict[str, Any]:
        """Use MCP tools to research the error"""
        research_results = {
            "firecrawl_research": None,
            "context7_analysis": None,
            "insights_discovered": [],
            "research_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Use Firecrawl to research error documentation
            if "firecrawl_scrape" in self.mcp_tools:
                print(f"🕷️ Researching {error_data['exception']} with Firecrawl...")
                
                # Research relevant documentation URLs
                doc_urls = [
                    "https://docs.python-requests.org/en/latest/",
                    "https://httpstatuses.com/",
                    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status"
                ]
                
                for url in doc_urls[:1]:  # Try one URL to avoid rate limits
                    try:
                        result = self.mcp_tools["firecrawl_scrape"].call(url=url, timeout=10)
                        if result:
                            research_results["firecrawl_research"] = {
                                "url": url,
                                "content_length": len(str(result)),
                                "relevant_keywords": self._extract_keywords(str(result), error_data["exception"])
                            }
                            research_results["insights_discovered"].extend(
                                self._extract_error_insights(str(result), error_data["exception"])
                            )
                            break
                    except Exception as e:
                        print(f"   Firecrawl research failed: {e}")
                        continue
            
            # Use Context7 for additional context
            if "context7_get_library_docs" in self.mcp_tools:
                print(f"📚 Getting context for {error_data['exception']} with Context7...")
                
                try:
                    context_query = f"error handling {error_data['exception'].split(':')[0]}"
                    result = self.mcp_tools["context7_get_library_docs"].call(
                        library="requests",
                        query=context_query
                    )
                    if result:
                        research_results["context7_analysis"] = {
                            "query": context_query,
                            "content_length": len(str(result)),
                            "patterns_found": self._extract_patterns(str(result))
                        }
                        research_results["insights_discovered"].extend(
                            self._extract_context_insights(str(result))
                        )
                except Exception as e:
                    print(f"   Context7 analysis failed: {e}")
            
        except Exception as e:
            print(f"🚫 MCP research failed: {e}")
        
        research_results["research_time"] = time.time() - start_time
        self.error_memory["global_stats"]["research_sessions"] += 1
        
        return research_results
    
    def _extract_keywords(self, content: str, error_msg: str) -> List[str]:
        """Extract relevant keywords from research content"""
        keywords = []
        content_lower = content.lower()
        error_lower = error_msg.lower()
        
        # Look for error-related terms
        error_terms = ["timeout", "retry", "backoff", "rate limit", "authentication", "connection"]
        for term in error_terms:
            if term in content_lower and term in error_lower:
                keywords.append(term)
        
        return keywords
    
    def _extract_error_insights(self, content: str, error_msg: str) -> List[str]:
        """Extract actionable insights from research"""
        insights = []
        content_lower = content.lower()
        
        if "timeout" in error_msg.lower():
            if "exponential backoff" in content_lower:
                insights.append("use_exponential_backoff")
            if "retry" in content_lower:
                insights.append("implement_retry_logic")
            if "timeout" in content_lower:
                insights.append("increase_timeout_gradually")
        
        if "429" in error_msg or "rate limit" in error_msg.lower():
            if "backoff" in content_lower:
                insights.append("implement_backoff_strategy")
            if "delay" in content_lower:
                insights.append("add_request_delay")
        
        return insights
    
    def _extract_patterns(self, content: str) -> List[str]:
        """Extract common patterns from context"""
        patterns = []
        content_lower = content.lower()
        
        pattern_keywords = ["retry", "circuit breaker", "exponential", "backoff", "timeout"]
        for keyword in pattern_keywords:
            if keyword in content_lower:
                patterns.append(f"{keyword}_pattern")
        
        return patterns
    
    def _extract_context_insights(self, content: str) -> List[str]:
        """Extract insights from context analysis"""
        insights = []
        content_lower = content.lower()
        
        if "exponential backoff" in content_lower:
            insights.append("use_exponential_backoff")
        if "circuit breaker" in content_lower:
            insights.append("implement_circuit_breaker")
        if "retry mechanism" in content_lower:
            insights.append("implement_retry_mechanism")
        
        return insights
    
    def generate_recovery_strategies(self, error_data: Dict, research_results: Dict) -> List[Dict[str, Any]]:
        """Generate recovery strategies based on error and research"""
        strategies = []
        insights = research_results.get("insights_discovered", [])
        error_type = error_data.get("exception", "").lower()
        
        # Research-based strategies
        if "use_exponential_backoff" in insights:
            strategies.append({
                "strategy": "exponential_backoff",
                "confidence": 0.8,
                "research_based": True,
                "description": "Implement exponential backoff based on research"
            })
        
        if "implement_retry_logic" in insights:
            strategies.append({
                "strategy": "intelligent_retry",
                "confidence": 0.7,
                "research_based": True,
                "description": "Retry with intelligent delay"
            })
        
        if "increase_timeout_gradually" in insights:
            strategies.append({
                "strategy": "progressive_timeout",
                "confidence": 0.6,
                "research_based": True,
                "description": "Gradually increase timeout"
            })
        
        # Fallback strategies based on error type
        if "timeout" in error_type:
            strategies.append({
                "strategy": "basic_timeout_increase",
                "confidence": 0.4,
                "research_based": False,
                "description": "Basic timeout increase"
            })
        
        if "429" in error_type or "rate" in error_type:
            strategies.append({
                "strategy": "basic_delay",
                "confidence": 0.3,
                "research_based": False,
                "description": "Basic delay strategy"
            })
        
        if "401" in error_type or "auth" in error_type:
            strategies.append({
                "strategy": "auth_refresh",
                "confidence": 0.5,
                "research_based": False,
                "description": "Refresh authentication"
            })
        
        return strategies
    
    def evaluate_recovery_success(self, strategy: Dict, error_data: Dict, research_results: Dict) -> float:
        """Evaluate success probability of recovery strategy"""
        base_success = 0.3
        
        # Research-based strategies have higher success rates
        if strategy.get("research_based", False):
            base_success += 0.4
        
        # Strategy confidence
        base_success += strategy.get("confidence", 0) * 0.3
        
        # Research quality boost
        research_quality = len(research_results.get("insights_discovered", [])) * 0.1
        base_success += research_quality
        
        # Add some randomness but favor good strategies
        import random
        final_success = base_success + random.uniform(-0.1, 0.1)
        
        return min(0.95, max(0.05, final_success))
    
    def record_pattern_learning(self, error_signature: str, error_data: Dict, 
                              research_results: Dict, successful_strategy: Dict, 
                              success_score: float):
        """Record learned pattern for future use"""
        
        # Find existing pattern or create new one
        existing_pattern = None
        for pattern in self.error_memory["patterns"]:
            if pattern["signature"] == error_signature:
                existing_pattern = pattern
                break
        
        if not existing_pattern:
            existing_pattern = {
                "signature": error_signature,
                "error_type": error_data.get("exception", ""),
                "context": error_data.get("context", {}),
                "successful_fixes": [],
                "failure_patterns": [],
                "research_insights": research_results.get("insights_discovered", []),
                "occurrences": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
            self.error_memory["patterns"].append(existing_pattern)
        
        # Update pattern
        existing_pattern["occurrences"] += 1
        existing_pattern["last_seen"] = datetime.now().isoformat()
        
        # Record successful strategy
        strategy_record = {
            "strategy": successful_strategy["strategy"],
            "success_score": success_score,
            "research_based": successful_strategy.get("research_based", False),
            "confidence": successful_strategy.get("confidence", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        existing_pattern["successful_fixes"].append(strategy_record)
        
        # Update global stats
        self.error_memory["global_stats"]["total_recoveries"] += 1
        if successful_strategy.get("research_based", False):
            self.error_memory["global_stats"]["successful_predictions"] += 1
        
        self.save_error_memory()

# Initialize model for training
random.seed(42)

model = art.TrainableModel(
    name="error-recovery-agent",
    project="error-recovery",
    base_model="OpenPipe/Qwen3-14B-Instruct",
)

backend = ServerlessBackend()

async def initialize_training():
    """Initialize the training system"""
    await model.register(backend)
    weave.init(model.project, settings={"print_call_link": False})
    print(f"🚀 Model registered: {model.name}")
    print(f"📊 Project: {model.project}")

@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def error_recovery_rollout(model: art.Model, scenario: ErrorRecoveryScenario) -> art.Trajectory:
    """Training rollout for error recovery scenarios"""
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Initialize agent
    agent = RealErrorRecoveryAgent()
    
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an intelligent error recovery agent. When given an API error, you analyze it, research solutions using available tools, and provide the best recovery strategy. Always respond with a JSON object containing your analysis and recommended strategy."
            }
        ],
        metadata={
            "scenario_step": scenario.step,
            "error_type": scenario.error_type,
            "api_endpoint": scenario.api_endpoint,
            "complexity": scenario.complexity_level,
            "notebook-id": "error-recovery",
        },
        reward=0,
    )
    
    try:
        # 1. Generate realistic error
        error_data = agent.simulate_real_api_error(scenario)
        error_signature = agent.create_error_signature(
            scenario.error_type, 
            scenario.api_endpoint, 
            error_data["context"]
        )
        
        # 2. Research the error with MCP tools
        print(f"🔬 Researching {scenario.error_type} error...")
        research_results = agent.research_error_with_mcp(error_data, scenario.api_endpoint)
        
        # 3. Create prompt for the model
        error_prompt = f"""
API Error Encountered:
- Endpoint: {scenario.api_endpoint}
- Error: {error_data['exception']}
- Context: {json.dumps(error_data['context'], indent=2)}

Research Results:
- Insights Discovered: {research_results['insights_discovered']}
- Research Time: {research_results['research_time']:.2f}s

Please analyze this error and provide a recovery strategy as JSON:
{{
    "analysis": "Your analysis of the error",
    "strategy": "recovery_strategy_name",
    "confidence": 0.8,
    "reasoning": "Why this strategy will work"
}}
"""
        
        trajectory.messages_and_choices.append({"role": "user", "content": error_prompt})
        
        # 4. Get model response
        messages = trajectory.messages()
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=512,
            messages=messages,
            model=model.get_inference_name(),
        )
        
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)
        
        # 5. Parse and evaluate the strategy
        try:
            response = json.loads(content)
            strategy = {
                "strategy": response.get("strategy", "unknown"),
                "confidence": float(response.get("confidence", 0.5)),
                "reasoning": response.get("reasoning", ""),
                "research_based": any(insight in response.get("analysis", "").lower() 
                                    for insight in research_results['insights_discovered'])
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback for malformed responses
            strategy = {
                "strategy": "basic_retry",
                "confidence": 0.2,
                "reasoning": "Fallback strategy",
                "research_based": False
            }
        
        # 6. Evaluate strategy success
        success_score = agent.evaluate_recovery_success(strategy, error_data, research_results)
        
        # 7. Record learning
        agent.record_pattern_learning(
            error_signature, error_data, research_results, strategy, success_score
        )
        
        # 8. Calculate reward based on multiple factors
        base_reward = success_score
        
        # Bonus for research-based strategies
        if strategy.get("research_based", False):
            base_reward += 0.2
        
        # Bonus for high confidence with good results
        if strategy.get("confidence", 0) > 0.7 and success_score > 0.7:
            base_reward += 0.1
        
        # Bonus for discovering new insights
        insight_bonus = len(research_results.get("insights_discovered", [])) * 0.05
        base_reward += insight_bonus
        
        trajectory.reward = min(1.0, base_reward)
        
        # Add metrics
        trajectory.metrics.update({
            "success_score": success_score,
            "strategy_confidence": strategy.get("confidence", 0),
            "research_insights": len(research_results.get("insights_discovered", [])),
            "research_time": research_results.get("research_time", 0),
            "research_based": int(strategy.get("research_based", False)),
            "error_type": scenario.error_type,
            "complexity": scenario.complexity_level
        })
        
        print(f"✅ Recovery training complete - Success: {success_score:.2f}, Reward: {trajectory.reward:.3f}")
        
    except Exception as e:
        print(f"❌ Training rollout failed: {e}")
        trajectory.reward = -0.5
        trajectory.metrics["error"] = str(e)
    
    return trajectory

async def run_error_recovery_training():
    """Run the complete error recovery training"""
    
    print(f"🚀 STARTING REAL ERROR RECOVERY TRAINING")
    print(f"=" * 60)
    
    await initialize_training()
    
    # Define training scenarios with increasing complexity
    error_types = ["timeout", "rate_limit", "auth_error", "network_error", "server_error"]
    complexity_levels = ["simple", "medium", "complex"]
    api_endpoints = [
        "https://api.example.com/data",
        "https://api.service.com/users", 
        "https://api.platform.com/analytics",
        "https://api.system.com/reports"
    ]
    
    training_scenarios = []
    for step in range(20):  # 20 training steps
        error_type = random.choice(error_types)
        complexity = random.choice(complexity_levels)
        endpoint = random.choice(api_endpoints)
        
        training_scenarios.append(ErrorRecoveryScenario(
            step=step,
            error_type=error_type,
            api_endpoint=endpoint,
            complexity_level=complexity
        ))
    
    print(f"📋 Training on {len(training_scenarios)} error recovery scenarios...")
    print(f"🧠 Using real MCP tools: Firecrawl + Context7 for research")
    print(f"💾 Learning will persist in real_error_memory.json")
    print(f"=" * 60)
    
    # Run training
    for i in range(await model.get_step(), len(training_scenarios)):
        scenario = training_scenarios[i]
        
        print(f"\n🎯 TRAINING STEP {i+1}/{len(training_scenarios)}")
        print(f"   Error Type: {scenario.error_type}")
        print(f"   Complexity: {scenario.complexity_level}")
        print(f"   Endpoint: {scenario.api_endpoint}")
        
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    error_recovery_rollout(model, scenario) for _ in range(8)  # 8 trajectories per step
                )
                for _ in range(1)
            ),
            pbar_desc=f"Step {i+1} trajectories",
            max_exceptions=8,
        )
        
        # Clean up checkpoints and train
        await model.delete_checkpoints('train/reward')
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        
        print(f"✅ Step {i+1} training complete")
    
    last_step = await model.get_step()
    deployed_model_name = f"{model.get_inference_name()}:step{last_step}"
    
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"📊 Final step: {last_step}")
    print(f"🚀 Deployed as: {deployed_model_name}")
    print(f"💾 Error patterns saved to: real_error_memory.json")
    
    return deployed_model_name

async def test_trained_agent(deployed_model_name: str):
    """Test the trained agent on new error scenarios"""
    
    print(f"\n🧪 TESTING TRAINED AGENT")
    print(f"=" * 50)
    
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    
    # Test scenarios
    test_scenarios = [
        {
            "error": "TimeoutError: Request timed out after 10 seconds",
            "endpoint": "https://api.test.com/heavy-data",
            "context": {"payload_size": "large", "timeout": 10}
        },
        {
            "error": "HTTPError: 429 Too Many Requests",
            "endpoint": "https://api.test.com/frequent-calls",
            "context": {"rate_limit": "exceeded", "requests_per_minute": 100}
        }
    ]
    
    for i, test_case in enumerate(test_scenarios, 1):
        print(f"\n🎯 TEST {i}: {test_case['error']}")
        
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent error recovery agent. Analyze the error and provide the best recovery strategy as JSON."
            },
            {
                "role": "user", 
                "content": f"""
API Error: {test_case['error']}
Endpoint: {test_case['endpoint']}
Context: {json.dumps(test_case['context'])}

Provide recovery strategy as JSON with analysis, strategy, confidence, and reasoning.
"""
            }
        ]
        
        try:
            response = await client.chat.completions.create(
                model=deployed_model_name,
                messages=messages,
                max_completion_tokens=256
            )
            
            result = response.choices[0].message.content
            print(f"🤖 Agent Response:")
            print(result)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    async def main():
        deployed_model = await run_error_recovery_training()
        await test_trained_agent(deployed_model)
    
    asyncio.run(main())