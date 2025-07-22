from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

from src.agents.analysis_aggregator_agent import AnalysisAggregatorAgent

class AgentState(TypedDict):
    text: str
    analysis: dict

def analyze_document_factory(llm):
    def analyze_document(state):
        """
        Analyzes the document using the AnalysisAggregatorAgent.

        Args:
            state: The current state of the graph.

        Returns:
            A dictionary with the analysis results.
        """
        text = state["text"]
        agent = AnalysisAggregatorAgent(llm)
        analysis = agent.analyze_document(text)
        return {"analysis": analysis}
    return analyze_document

def create_workflow(llm):
    """
    Creates the LangGraph workflow.

    Returns:
        The compiled LangGraph app.
    """
    workflow = StateGraph(AgentState)

    analyze_document_node = analyze_document_factory(llm)
    workflow.add_node("analyze_document", analyze_document_node)
    workflow.set_entry_point("analyze_document")
    workflow.add_edge("analyze_document", END)

    return workflow.compile()
