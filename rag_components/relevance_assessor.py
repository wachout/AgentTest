from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from .llm_client import get_llm
from .state import AgentState

def assess_relevance(state: AgentState):
    """
    Assesses the relevance of retrieved documents.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a relevance assessor. Your task is to determine if the retrieved documents are relevant to the user's query. Respond in JSON format with 'is_relevant' (boolean) and 'reasoning' (string). Query: {query}\n\nDocuments:\n{documents}"
    )
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    doc_content = "\n".join([doc["content"] for doc in state["retrieved_docs"]])
    relevance = chain.invoke({"query": state["enhanced_query"], "documents": doc_content})

    return {"document_relevance": relevance}