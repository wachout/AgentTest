from src.agent import create_agent

def test_agent_knowledge_base_flow():
    """
    Tests the full agent flow for a knowledge base query.
    """
    agent = create_agent()
    query = "What is LangGraph?"
    inputs = {"query": query}
    response = agent.invoke(inputs)
    assert "Searching the knowledge base for" in response["result"]
