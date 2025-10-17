from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from rag_components.state import AgentState
from rag_components.query_enhancer import enhance_query
from rag_components.topic_validator import validate_topic
from rag_components.content_retriever import retrieve_content
from rag_components.relevance_assessor import assess_relevance
from rag_components.response_generator import generate_response
from rag_components.query_optimizer import optimize_query

def should_retrieve(state: AgentState):
    if state["topic_classification"]["classification"] == "out_of_domain":
        return "end"
    return "retrieve_content"

def is_relevant(state: AgentState):
    if state["document_relevance"]["is_relevant"]:
        return "generate_response"
    elif state["refinement_attempts"] >= 2:
        return "end"
    else:
        return "optimize_query"

def create_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("enhance_query", enhance_query)
    workflow.add_node("validate_topic", validate_topic)
    workflow.add_node("retrieve_content", retrieve_content)
    workflow.add_node("assess_relevance", assess_relevance)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("optimize_query", optimize_query)

    # Add edges
    workflow.set_entry_point("enhance_query")
    workflow.add_edge("enhance_query", "validate_topic")
    workflow.add_conditional_edges("validate_topic", should_retrieve, {
        "retrieve_content": "retrieve_content",
        "end": END
    })
    workflow.add_edge("retrieve_content", "assess_relevance")
    workflow.add_conditional_edges("assess_relevance", is_relevant, {
        "generate_response": "generate_response",
        "optimize_query": "optimize_query",
        "end": END
    })
    workflow.add_edge("generate_response", END)
    workflow.add_edge("optimize_query", "retrieve_content") # Loop back to retrieve

    return workflow.compile()

if __name__ == "__main__":
    try:
        app = create_graph()

        # Initial question
        inputs = {
            "chat_history": [],
            "question": "how do I reset my password?",
            "refinement_attempts": 0
        }
        result = app.invoke(inputs)
        print("Final Response:", result.get("generation", "I could not find a relevant answer."))

        # Follow-up question
        inputs = {
            "chat_history": [HumanMessage(content="how do I reset my password?"), result.get("generation")],
            "question": "what about billing?",
            "refinement_attempts": 0
        }
        result = app.invoke(inputs)
        print("Final Response:", result.get("generation", "I could not find a relevant answer."))

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have set your API keys in a .env file.")