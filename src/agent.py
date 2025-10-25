import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.tools import search_knowledge_base, query_database, perform_data_analysis

# Load environment variables from .env file
load_dotenv()


# 1. Agent State
class AgentState(TypedDict, total=False):
    query: str
    intent: Literal["KNOWLEDGE_BASE_QUERY", "DATABASE_REPORT_REQUEST", "DATA_ANALYSIS_REQUEST"]
    result: str


# 2. Nodes
def get_intent_node(state: AgentState):
    """
    Identifies the user's intent based on their query.
    """
    query = state["query"]
    # Initialize the DeepSeek LLM
    llm = ChatOpenAI(
        temperature=0.6,
        model="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    # Create a prompt template for intent classification
    prompt_template = """
    Your task is to analyze the user's query and classify it into one of the following three categories:

    1.  **KNOWLEDGE_BASE_QUERY**: The user is asking a question that can be answered from a knowledge base.
        - *Examples*: "What is LangGraph?", "How does the RAG model work?"

    2.  **DATABASE_REPORT_REQUEST**: The user is requesting a statistical report or data from a database.
        - *Examples*: "Show me the sales figures for the last quarter.", "What are the top-selling products?"

    3.  **DATA_ANALYSIS_REQUEST**: The user is asking for an in-depth analysis of data, which may involve generating reports and making decisions.
        - *Examples*: "Analyze the customer churn rate and suggest strategies to reduce it.", "Provide a detailed market analysis for our new product line."

    Based on the query provided, which category does it fall into? Return only the category name.

    **Query**: "{query}"

    **Classification**:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

    # Create the intent recognition chain
    chain = prompt | llm | StrOutputParser()

    # Get the intent
    intent = chain.invoke({"query": query})
    return {"intent": intent.strip()}


def route_intent(state: AgentState):
    """
    Routes to the correct tool based on the intent.
    """
    intent = state["intent"]
    if intent == "KNOWLEDGE_BASE_QUERY":
        return "knowledge_base"
    elif intent == "DATABASE_REPORT_REQUEST":
        return "database_report"
    elif intent == "DATA_ANALYSIS_REQUEST":
        return "data_analysis"
    else:
        return END


def execute_knowledge_base_query(state: AgentState):
    query = state["query"]
    result = search_knowledge_base.invoke({"query": query})
    return {"result": result}


def execute_database_report_request(state: AgentState):
    query = state["query"]
    result = query_database.invoke({"query": query})
    return {"result": result}


def execute_data_analysis_request(state: AgentState):
    query = state["query"]
    result = perform_data_analysis.invoke({"query": query})
    return {"result": result}


# 3. Graph Definition
def create_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("get_intent", get_intent_node)
    workflow.add_node("knowledge_base", execute_knowledge_base_query)
    workflow.add_node("database_report", execute_database_report_request)
    workflow.add_node("data_analysis", execute_data_analysis_request)

    workflow.set_entry_point("get_intent")

    workflow.add_conditional_edges(
        "get_intent",
        route_intent,
        {
            "knowledge_base": "knowledge_base",
            "database_report": "database_report",
            "data_analysis": "data_analysis",
        },
    )

    workflow.add_edge("knowledge_base", END)
    workflow.add_edge("database_report", END)
    workflow.add_edge("data_analysis", END)

    return workflow.compile()
