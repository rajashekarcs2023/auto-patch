"""
Research Environment with Firecrawl Integration
"""
import asyncio
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from firecrawl import FirecrawlApp
from context_engine import ContextManager, ContextChunk


@dataclass
class ResearchTask:
    """A research task for the agent to complete"""
    id: str
    question: str
    task_type: str  # "regulation", "technical", "business", etc.
    expected_sources: List[str]  # What types of sources should be useful
    difficulty: float  # 0.0 to 1.0


class ResearchScenario(BaseModel):
    """Scenario for RL training"""
    step: int
    task: ResearchTask
    available_context: List[str]  # Context chunk IDs available for this task


class ResearchEnvironment:
    """Environment that provides research tasks and evaluates performance"""
    
    def __init__(self, firecrawl_api_key: str):
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key) if firecrawl_api_key else None
        self.context_manager = ContextManager()
        self._setup_initial_context()
    
    def _setup_initial_context(self):
        """Set up initial context pool with diverse content"""
        
        # Simulated context chunks (in real version, these would come from Firecrawl)
        initial_contexts = [
            {
                "content": "Autonomous vehicle regulations vary significantly by jurisdiction. In the US, the NHTSA provides federal guidelines while states like California and Texas have their own specific requirements for testing and deployment.",
                "source": "web",
                "id": "av_regulations_us"
            },
            {
                "content": "OAuth2 is an authorization framework that enables applications to obtain limited access to user accounts. The flow involves authorization grants, access tokens, and refresh tokens.",
                "source": "docs", 
                "id": "oauth2_basics"
            },
            {
                "content": "Recent studies show that 73% of companies struggle with AI governance frameworks, particularly around model transparency and bias detection.",
                "source": "web",
                "id": "ai_governance_stats"
            },
            {
                "content": "Python implementation of OAuth2 requires libraries like requests-oauthlib. Key steps: register app, redirect user, exchange code for token, make authenticated requests.",
                "source": "docs",
                "id": "oauth2_python_impl"
            },
            {
                "content": "The European Union's AI Act, passed in 2024, establishes comprehensive regulations for AI systems with risk-based classifications from minimal to unacceptable risk.",
                "source": "web", 
                "id": "eu_ai_act_2024"
            },
            {
                "content": "Machine learning model evaluation should include accuracy, precision, recall, F1-score, and fairness metrics across different demographic groups.",
                "source": "memory",
                "id": "ml_evaluation_best_practices"
            },
            {
                "content": "Venture capital funding for AI startups reached $25.2B in 2024, with a focus on enterprise AI solutions and autonomous systems.",
                "source": "web",
                "id": "ai_vc_funding_2024"
            },
            {
                "content": "RESTful API design principles: use HTTP methods correctly, implement proper status codes, version your APIs, and provide comprehensive documentation.",
                "source": "docs",
                "id": "rest_api_principles"
            }
        ]
        
        for ctx in initial_contexts:
            self.context_manager.add_context(
                content=ctx["content"],
                source=ctx["source"],
                chunk_id=ctx["id"]
            )
    
    async def research_with_firecrawl(self, query: str, max_results: int = 3) -> List[str]:
        """Use Firecrawl to get fresh research content (fallback to mock if API unavailable)"""
        if not self.firecrawl:
            # Mock research results
            mock_results = [
                f"Recent research on '{query}' shows emerging trends in the field with significant implications for industry applications.",
                f"Studies indicate that {query} is becoming increasingly important with new regulatory frameworks being developed.",
                f"Industry experts suggest that {query} will see major developments in the next 2-3 years based on current technological advances."
            ]
            return mock_results[:max_results]
        
        try:
            # Use Firecrawl search (simplified for demo)
            search_result = await asyncio.to_thread(
                self.firecrawl.search,
                query=query,
                limit=max_results
            )
            
            if search_result and 'data' in search_result:
                return [item.get('content', '')[:500] for item in search_result['data'][:max_results]]
            else:
                return [f"No specific results found for {query}, using cached knowledge base."]
                
        except Exception as e:
            print(f"Firecrawl error: {e}, using mock data")
            return [f"Research indicates {query} is an active area of development with ongoing industry investment."]
    
    def get_research_tasks(self) -> List[ResearchTask]:
        """Get predefined research tasks for training/demo"""
        return [
            ResearchTask(
                id="av_regulations",
                question="What are the current regulations for autonomous vehicle testing in the United States?",
                task_type="regulation",
                expected_sources=["web", "docs"],
                difficulty=0.6
            ),
            ResearchTask(
                id="oauth2_implementation", 
                question="How do you implement OAuth2 authentication in a Python web application?",
                task_type="technical",
                expected_sources=["docs", "memory"],
                difficulty=0.4
            ),
            ResearchTask(
                id="ai_governance",
                question="What are the key challenges companies face when implementing AI governance frameworks?",
                task_type="business",
                expected_sources=["web", "memory"],
                difficulty=0.7
            ),
            ResearchTask(
                id="eu_ai_act",
                question="What are the main requirements of the European Union's AI Act for high-risk AI systems?",
                task_type="regulation", 
                expected_sources=["web", "docs"],
                difficulty=0.8
            ),
            ResearchTask(
                id="ml_evaluation",
                question="What metrics should be used to evaluate machine learning models for fairness and bias?",
                task_type="technical",
                expected_sources=["memory", "docs"],
                difficulty=0.5
            )
        ]
    
    async def add_live_context(self, task: ResearchTask) -> List[str]:
        """Add fresh context from Firecrawl for a task"""
        new_content = await self.research_with_firecrawl(task.question, max_results=2)
        
        added_ids = []
        for i, content in enumerate(new_content):
            chunk_id = f"live_{task.id}_{i}"
            self.context_manager.add_context(
                content=content,
                source="web",
                chunk_id=chunk_id
            )
            added_ids.append(chunk_id)
        
        return added_ids
    
    def evaluate_research_quality(self, research_output: str, task: ResearchTask) -> Dict[str, float]:
        """Evaluate quality of research output"""
        # Simplified evaluation (in real version, would use LLM judge)
        output_lower = research_output.lower()
        question_lower = task.question.lower()
        
        # Basic relevance scoring
        relevance_score = 0.0
        key_terms = question_lower.split()
        for term in key_terms:
            if len(term) > 3 and term in output_lower:
                relevance_score += 0.2
        
        relevance_score = min(relevance_score, 1.0)
        
        # Length and detail scoring
        detail_score = min(len(research_output) / 500, 1.0)  # Normalize to 500 chars
        
        # Source diversity (if output mentions different types of sources)
        source_indicators = ["study", "regulation", "documentation", "research", "industry", "official"]
        source_diversity = sum(1 for indicator in source_indicators if indicator in output_lower) / len(source_indicators)
        
        overall_score = (relevance_score * 0.5 + detail_score * 0.3 + source_diversity * 0.2)
        
        return {
            "relevance": relevance_score,
            "detail": detail_score, 
            "source_diversity": source_diversity,
            "overall": overall_score
        }
    
    def get_context_efficiency_score(self, context_metrics: Dict[str, float]) -> float:
        """Calculate efficiency score based on context usage"""
        selection_ratio = context_metrics.get("selection_ratio", 1.0)
        efficiency_score = context_metrics.get("efficiency_score", 0.0)
        
        # Reward for using less context while maintaining quality
        efficiency_bonus = (1.0 - selection_ratio) * 0.5 + efficiency_score * 0.5
        return efficiency_bonus
    
    def create_scenario(self, step: int, task_id: Optional[str] = None) -> ResearchScenario:
        """Create a research scenario for training"""
        tasks = self.get_research_tasks()
        
        if task_id:
            task = next((t for t in tasks if t.id == task_id), tasks[0])
        else:
            task = random.choice(tasks)
        
        # Get available context chunk IDs
        available_context = [chunk.id for chunk in self.context_manager.context_pool]
        
        return ResearchScenario(
            step=step,
            task=task,
            available_context=available_context
        )