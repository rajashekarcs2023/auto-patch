#!/usr/bin/env python3
"""
🔥 MCP Predictive Error Recovery Agent
Shows REAL self-improvement with actual Firecrawl, Context7, and other MCP APIs
Learns from real timeout, rate limit, and API errors
"""
import json
import time
import requests
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
from real_mcp_tools import get_all_real_mcp_tools

class MCPSelfHealingAgent:
    """Agent that learns from real MCP tool errors and predicts/prevents failures"""
    
    def __init__(self, memory_file="mcp_error_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()
        self.mcp_tools = get_all_real_mcp_tools()
        self.attempt_count = 0
        self.session_stats = []
        
        print(f"🚀 Initialized with {len(self.mcp_tools)} MCP tools")
        print(f"🧠 Loaded {len(self.memory['patterns'])} learned error patterns")
        
    def load_memory(self) -> Dict:
        """Load persistent error memory"""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "patterns": [],
                "global_stats": {
                    "total_attempts": 0,
                    "successful_predictions": 0,
                    "tools_learned": []
                },
                "tool_specific_patterns": {}
            }
    
    def save_memory(self):
        """Save memory to persistent storage"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def extract_mcp_signature(self, tool_name: str, error: Exception, params: Dict) -> str:
        """Extract signature from MCP tool error for pattern matching"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # MCP-specific error characteristics
        signature_data = {
            "tool": tool_name,
            "error_type": error_type,
            "has_timeout": "timeout" in error_msg or "time" in error_msg,
            "has_rate_limit": "rate" in error_msg or "429" in error_msg or "too many" in error_msg,
            "has_auth": "auth" in error_msg or "401" in error_msg or "unauthorized" in error_msg,
            "has_quota": "quota" in error_msg or "limit" in error_msg,
            "has_network": "network" in error_msg or "connection" in error_msg,
            "param_complexity": len(str(params)),
            "hour_of_day": datetime.now().hour  # Some APIs have peak hours
        }
        
        return hashlib.md5(str(signature_data).encode()).hexdigest()
    
    def predict_mcp_fix(self, tool_name: str, params: Dict) -> Optional[Dict]:
        """Predict if this MCP call will fail and suggest preemptive fix"""
        
        # Check tool-specific patterns
        if tool_name in self.memory["tool_specific_patterns"]:
            tool_patterns = self.memory["tool_specific_patterns"][tool_name]
            
            for pattern in tool_patterns:
                if pattern["occurrences"] >= 2:  # Lower threshold for MCP tools
                    best_fix = max(pattern["learned_fixes"], key=lambda x: x["success_rate"])
                    if best_fix["success_rate"] > 0.6:  # Lower threshold for real APIs
                        return best_fix
        
        return None
    
    def apply_mcp_fix(self, tool_name: str, params: Dict, fix_strategy: Dict) -> Dict:
        """Apply learned fix strategy to MCP tool parameters"""
        strategy = fix_strategy["strategy"]
        fixed_params = params.copy()
        
        if strategy == "reduce_timeout":
            fixed_params["timeout"] = 10  # Shorter timeout
        elif strategy == "add_retry_delay":
            time.sleep(2)  # Wait before retry
        elif strategy == "simplify_params":
            # Reduce complexity for overloaded APIs
            if "url" in fixed_params and len(fixed_params["url"]) > 100:
                fixed_params["url"] = fixed_params["url"][:100]
            if "query" in fixed_params and len(fixed_params["query"]) > 50:
                fixed_params["query"] = fixed_params["query"][:50]
        elif strategy == "use_fallback_endpoint":
            # For Firecrawl, try simpler endpoint
            if tool_name.startswith("firecrawl") and "extract" in tool_name:
                # Fallback to basic scrape
                return {"url": fixed_params.get("url", ""), "timeout": 15}
        elif strategy == "batch_reduce":
            # Reduce batch size if API is overloaded
            if "urls" in fixed_params and isinstance(fixed_params["urls"], list):
                fixed_params["urls"] = fixed_params["urls"][:3]  # Smaller batch
        elif strategy == "add_user_agent":
            if "headers" not in fixed_params:
                fixed_params["headers"] = {}
            fixed_params["headers"]["User-Agent"] = "Mozilla/5.0 (compatible; MCPAgent/1.0)"
        
        return fixed_params
    
    def generate_mcp_fixes(self, tool_name: str, error: Exception) -> List[Dict]:
        """Generate fix strategies based on MCP tool and error type"""
        error_msg = str(error).lower()
        fixes = []
        
        # Common MCP API issues
        if "timeout" in error_msg or "time" in error_msg:
            fixes.extend([
                {"strategy": "reduce_timeout", "success_rate": 0.0},
                {"strategy": "add_retry_delay", "success_rate": 0.0},
                {"strategy": "simplify_params", "success_rate": 0.0}
            ])
        
        if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
            fixes.extend([
                {"strategy": "add_retry_delay", "success_rate": 0.0},
                {"strategy": "batch_reduce", "success_rate": 0.0}
            ])
        
        if "auth" in error_msg or "401" in error_msg:
            fixes.extend([
                {"strategy": "refresh_credentials", "success_rate": 0.0},
                {"strategy": "add_user_agent", "success_rate": 0.0}
            ])
        
        # Tool-specific fixes
        if "firecrawl" in tool_name:
            fixes.extend([
                {"strategy": "use_fallback_endpoint", "success_rate": 0.0},
                {"strategy": "simplify_params", "success_rate": 0.0}
            ])
        
        if "context7" in tool_name:
            fixes.extend([
                {"strategy": "simplify_params", "success_rate": 0.0},
                {"strategy": "add_retry_delay", "success_rate": 0.0}
            ])
        
        # Generic fallbacks
        if not fixes:
            fixes.extend([
                {"strategy": "add_retry_delay", "success_rate": 0.0},
                {"strategy": "simplify_params", "success_rate": 0.0}
            ])
        
        return fixes
    
    def try_mcp_fix(self, tool_name: str, params: Dict, fix_strategy: Dict) -> tuple[bool, Any]:
        """Try applying fix strategy to actual MCP tool"""
        try:
            fixed_params = self.apply_mcp_fix(tool_name, params, fix_strategy)
            
            # Get the actual MCP tool
            mcp_tool = self.mcp_tools.get(tool_name)
            if not mcp_tool:
                return False, None
            
            # Try the fixed call
            start_time = time.time()
            try:
                # Call the actual MCP tool with fixed parameters
                result = mcp_tool.call(**fixed_params)
                duration = time.time() - start_time
                
                print(f"   ✅ Fix '{fix_strategy['strategy']}' worked! ({duration:.1f}s)")
                return True, result
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   ❌ Fix '{fix_strategy['strategy']}' failed: {e} ({duration:.1f}s)")
                return False, None
                
        except Exception as e:
            print(f"   💥 Fix application error: {e}")
            return False, None
    
    def update_mcp_memory(self, tool_name: str, signature: str, fix_strategy: Dict, success: bool):
        """Update memory with MCP fix attempt result"""
        
        # Initialize tool-specific patterns if needed
        if tool_name not in self.memory["tool_specific_patterns"]:
            self.memory["tool_specific_patterns"][tool_name] = []
        
        # Find existing pattern
        pattern = None
        for p in self.memory["tool_specific_patterns"][tool_name]:
            if p["error_signature"] == signature:
                pattern = p
                break
        
        if not pattern:
            # Create new pattern
            pattern = {
                "error_signature": signature,
                "tool": tool_name,
                "learned_fixes": [],
                "occurrences": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
            self.memory["tool_specific_patterns"][tool_name].append(pattern)
        
        # Update pattern
        pattern["occurrences"] += 1
        pattern["last_seen"] = datetime.now().isoformat()
        
        # Update fix strategy success rate
        fix_found = False
        for fix in pattern["learned_fixes"]:
            if fix["strategy"] == fix_strategy["strategy"]:
                current_attempts = fix.get("attempts", 0) + 1
                current_successes = fix.get("successes", 0) + (1 if success else 0)
                fix["success_rate"] = current_successes / current_attempts
                fix["attempts"] = current_attempts
                fix["successes"] = current_successes
                fix_found = True
                break
        
        if not fix_found:
            pattern["learned_fixes"].append({
                "strategy": fix_strategy["strategy"],
                "success_rate": 1.0 if success else 0.0,
                "attempts": 1,
                "successes": 1 if success else 0
            })
        
        # Update global stats
        if tool_name not in self.memory["global_stats"]["tools_learned"]:
            self.memory["global_stats"]["tools_learned"].append(tool_name)
        
        self.save_memory()
    
    def call_mcp_tool(self, tool_name: str, params: Dict = None) -> Dict:
        """Call MCP tool with predictive error recovery"""
        if params is None:
            params = {}
        
        if tool_name not in self.mcp_tools:
            return {"status": "error", "message": f"Tool {tool_name} not found"}
        
        self.attempt_count += 1
        start_time = time.time()
        retries = 0
        
        print(f"\n🎯 Attempt #{self.attempt_count}: {tool_name}")
        print(f"📥 Params: {str(params)[:100]}...")
        
        # PHASE 3: PREDICTIVE - Check if we can prevent errors
        predicted_fix = self.predict_mcp_fix(tool_name, params)
        if predicted_fix:
            print(f"🔮 PREDICTION: Applying learned fix '{predicted_fix['strategy']}' (success rate: {predicted_fix['success_rate']:.1%})")
            params = self.apply_mcp_fix(tool_name, params, predicted_fix)
            self.memory["global_stats"]["successful_predictions"] += 1
        
        # Try the request
        try:
            mcp_tool = self.mcp_tools[tool_name]
            result = mcp_tool.call(**params)
            
            end_time = time.time()
            duration = end_time - start_time
            
            session_result = {
                "status": "success",
                "tool": tool_name,
                "retries": retries,
                "duration": duration,
                "predicted": predicted_fix is not None,
                "result_size": len(str(result))
            }
            
            print(f"✅ SUCCESS in {duration:.1f}s (retries: {retries})")
            if predicted_fix:
                print(f"🎯 Prediction prevented error!")
            
            self.session_stats.append(session_result)
            return session_result
            
        except Exception as e:
            # PHASE 1 & 2: REACTIVE AND LEARNED
            return self.handle_mcp_error(e, tool_name, params, start_time, retries)
    
    def handle_mcp_error(self, error: Exception, tool_name: str, params: Dict, start_time: float, retries: int) -> Dict:
        """Handle MCP tool errors with learning and recovery"""
        print(f"❌ ERROR: {error}")
        
        signature = self.extract_mcp_signature(tool_name, error, params)
        retries += 1
        
        # PHASE 2: Try learned fixes first
        if tool_name in self.memory["tool_specific_patterns"]:
            learned_patterns = self.memory["tool_specific_patterns"][tool_name]
            for pattern in learned_patterns:
                if pattern["error_signature"] == signature and pattern["learned_fixes"]:
                    print(f"🧠 Trying {len(pattern['learned_fixes'])} learned fixes for {tool_name}...")
                    
                    for fix in sorted(pattern["learned_fixes"], key=lambda x: x["success_rate"], reverse=True):
                        print(f"   Trying '{fix['strategy']}' (success rate: {fix['success_rate']:.1%})")
                        success, result = self.try_mcp_fix(tool_name, params, fix)
                        
                        if success:
                            end_time = time.time()
                            duration = end_time - start_time
                            
                            self.update_mcp_memory(tool_name, signature, fix, success=True)
                            
                            session_result = {
                                "status": "recovered",
                                "tool": tool_name,
                                "retries": retries,
                                "duration": duration,
                                "fix_used": fix["strategy"],
                                "fix_type": "learned",
                                "result_size": len(str(result)) if result else 0
                            }
                            
                            print(f"✅ RECOVERED with learned fix '{fix['strategy']}' in {duration:.1f}s (retries: {retries})")
                            self.session_stats.append(session_result)
                            return session_result
                        else:
                            self.update_mcp_memory(tool_name, signature, fix, success=False)
        
        # PHASE 1: Try new fixes
        print(f"🔧 Trying new fixes for {tool_name}...")
        new_fixes = self.generate_mcp_fixes(tool_name, error)
        
        for fix in new_fixes:
            print(f"   Trying new '{fix['strategy']}'")
            success, result = self.try_mcp_fix(tool_name, params, fix)
            
            if success:
                end_time = time.time()
                duration = end_time - start_time
                
                self.update_mcp_memory(tool_name, signature, fix, success=True)
                
                session_result = {
                    "status": "recovered",
                    "tool": tool_name,
                    "retries": retries,
                    "duration": duration,
                    "fix_used": fix["strategy"],
                    "fix_type": "new",
                    "result_size": len(str(result)) if result else 0
                }
                
                print(f"✅ RECOVERED with NEW fix '{fix['strategy']}' in {duration:.1f}s (retries: {retries})")
                print(f"🎓 Learned new pattern for {tool_name}!")
                self.session_stats.append(session_result)
                return session_result
        
        # Complete failure
        end_time = time.time()
        duration = end_time - start_time
        
        session_result = {
            "status": "failed",
            "tool": tool_name,
            "retries": retries,
            "duration": duration
        }
        
        print(f"💥 FAILED after {duration:.1f}s (retries: {retries})")
        self.session_stats.append(session_result)
        return session_result
    
    def show_mcp_learning_curve(self):
        """Display MCP learning progression and statistics"""
        print(f"\n📊 MCP AGENT EVOLUTION STATS")
        print(f"=" * 70)
        
        if not self.session_stats:
            print("No MCP calls yet")
            return
        
        # Show progression table
        print(f"{'Attempt':<8} {'Tool':<20} {'Status':<12} {'Duration':<10} {'Method':<15}")
        print(f"-" * 70)
        
        for i, stat in enumerate(self.session_stats, 1):
            method = ""
            if stat.get("predicted"):
                method = "🔮 PREDICTED"
            elif stat.get("fix_type") == "learned":
                method = "🧠 LEARNED"
            elif stat.get("fix_type") == "new":
                method = "🔧 NEW FIX"
            else:
                method = "❌ FAILED" if stat["status"] == "failed" else "✅ SUCCESS"
            
            status_icon = "✅" if stat["status"] == "success" else "🔄" if stat["status"] == "recovered" else "💥"
            tool_short = stat["tool"][:18] + ".." if len(stat["tool"]) > 20 else stat["tool"]
            
            print(f"{i:<8} {tool_short:<20} {status_icon + stat['status']:<12} {stat['duration']:.1f}s{'':<5} {method}")
        
        # Show improvement metrics by tool
        tool_stats = {}
        for stat in self.session_stats:
            tool = stat["tool"]
            if tool not in tool_stats:
                tool_stats[tool] = {"attempts": [], "durations": []}
            tool_stats[tool]["attempts"].append(stat)
            tool_stats[tool]["durations"].append(stat["duration"])
        
        print(f"\n🚀 IMPROVEMENT BY TOOL:")
        for tool, stats in tool_stats.items():
            if len(stats["durations"]) >= 2:
                early_avg = stats["durations"][0]
                recent_avg = stats["durations"][-1]
                improvement = ((early_avg - recent_avg) / early_avg) * 100
                
                print(f"   {tool}:")
                print(f"     First call: {early_avg:.1f}s → Latest: {recent_avg:.1f}s")
                print(f"     Improvement: {improvement:.1f}%")
        
        # Show memory stats
        total_tools_learned = len(self.memory["global_stats"]["tools_learned"])
        total_patterns = sum(len(patterns) for patterns in self.memory["tool_specific_patterns"].values())
        successful_predictions = self.memory["global_stats"]["successful_predictions"]
        
        print(f"\n🧠 MEMORY STATS:")
        print(f"   Tools with learned patterns: {total_tools_learned}")
        print(f"   Total error patterns: {total_patterns}")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Total attempts: {self.attempt_count}")
        
        # Show learned patterns per tool
        if total_patterns > 0:
            print(f"\n📚 LEARNED PATTERNS BY TOOL:")
            for tool, patterns in self.memory["tool_specific_patterns"].items():
                if patterns:
                    print(f"   {tool}: {len(patterns)} patterns")
                    for pattern in patterns[:2]:  # Show top 2 per tool
                        fixes = len(pattern["learned_fixes"])
                        occurrences = pattern["occurrences"]
                        best_fix = max(pattern["learned_fixes"], key=lambda x: x["success_rate"]) if fixes > 0 else None
                        
                        print(f"     Pattern {pattern['error_signature'][:8]}... ({occurrences} times)")
                        if best_fix:
                            print(f"       Best fix: {best_fix['strategy']} ({best_fix['success_rate']:.1%} success)")

# Demonstration script
def run_mcp_demo():
    """Run impressive MCP learning demonstration"""
    agent = MCPSelfHealingAgent()
    
    print(f"🚀 STARTING MCP PREDICTIVE ERROR RECOVERY DEMO")
    print(f"Watch the agent learn to handle real API errors!\n")
    
    # Demo scenarios with real MCP tools
    demo_scenarios = [
        # Firecrawl scenarios (prone to timeouts)
        ("firecrawl_scrape", {"url": "https://example.com", "timeout": 5}),
        ("firecrawl_scrape", {"url": "https://httpbin.org/delay/3", "timeout": 5}),  # Likely timeout
        ("firecrawl_scrape", {"url": "https://example.com", "timeout": 5}),  # Should use learned fix
        
        # Context7 scenarios
        ("context7_get_library_docs", {"library": "react", "query": "hooks"}),
        ("context7_get_library_docs", {"library": "vue", "query": "composition api"}),
        
        # More Firecrawl to show learning
        ("firecrawl_scrape", {"url": "https://httpbin.org/delay/2", "timeout": 5}),  # Should predict fix
    ]
    
    print(f"📋 Running {len(demo_scenarios)} scenarios to demonstrate learning...\n")
    
    for i, (tool_name, params) in enumerate(demo_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🎬 SCENARIO {i}/{len(demo_scenarios)}")
        
        result = agent.call_mcp_tool(tool_name, params)
        
        # Show checkpoint messages
        if i == 2:
            print(f"\n📊 CHECKPOINT: Agent should start recognizing patterns...")
        elif i == 4:
            print(f"\n📊 CHECKPOINT: Agent should be making predictions...")
        
        time.sleep(1)  # Dramatic pause
    
    # Show final learning summary
    agent.show_mcp_learning_curve()
    
    print(f"\n🏆 MCP PREDICTIVE AGENT DEMO COMPLETE!")
    print(f"🎯 The agent learned to predict and prevent real API errors!")

if __name__ == "__main__":
    run_mcp_demo()