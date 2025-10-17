from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm
from .state import AgentState

def optimize_query(state: AgentState):
    """
    Optimizes the query for better search results.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a query optimizer. Your task is to refine the user's query to improve search results. Generate a new, more specific query based on the original. Original Query: {query}"
    )
    chain = prompt | llm | StrOutputParser()

    optimized_query = chain.invoke({"query": state["enhanced_query"]})

    return {"enhanced_query": optimized_query, "refinement_attempts": state["refinement_attempts"] + 1}