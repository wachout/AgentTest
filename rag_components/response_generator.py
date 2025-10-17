from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm
from .state import AgentState

def generate_response(state: AgentState):
    """
    Generates a response based on the context.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Generate a concise and informative answer to the user's query based on the provided context. If the context is not relevant, say so. Query: {query}\n\nContext:\n{context}"
    )
    chain = prompt | llm | StrOutputParser()

    context_str = "\n".join([f"Title: {doc['title']}\nContent: {doc['content']}" for doc in state["retrieved_docs"]])
    generation = chain.invoke({"query": state["enhanced_query"], "context": context_str})

    return {"generation": generation}