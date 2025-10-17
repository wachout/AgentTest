from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Defines the state for the RAG agent graph.
    """
    chat_history: Optional[List[BaseMessage]]
    question: str
    enhanced_query: str
    topic_classification: dict
    retrieved_docs: List[dict]
    document_relevance: dict
    generation: str
    refinement_attempts: int