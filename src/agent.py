import os
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from dotenv import load_dotenv

from lightrag.core.embedder import Embedder
from lightrag.components.model_client import TransformersClient
from lightrag.components.data_process.text_splitter import TextSplitter
from lightrag.core.types import Document
import numpy as np

# --- Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

# --- LangGraph State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    original_query: str
    rewritten_query: str
    documents: List[dict]
    expanded_documents: List[dict]
    reranked_documents: List[dict]
    generation: str
    cited_generation: str

from .vector_store import VectorStore
import re

# --- Global Variables ---
vector_store = None
embedder = None
neo4j_graph = None

def initialize_retriever():
    """
    Initializes the vector store and embedder.
    """
    global vector_store, embedder
    if vector_store is None:
        vector_store = VectorStore()

    if embedder is None:
        client = TransformersClient(model_name="thenlper/gte-base")
        embedder = Embedder(model_client=client)

# --- Tools ---
@tool
def rewrite_query(messages: List[BaseMessage]):
    """Rewrites the user's query for better retrieval."""
    last_message = messages[-1]
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found. Using original query.")
        return {"rewritten_query": last_message.content}

    prompt = f"""
    Rewrite the following user query to be more effective for a retrieval-augmented generation system.
    The rewritten query should be optimized to find relevant documents about "The Wandering Earth".

    Original Query: {last_message.content}

    Rewritten Query:
    """
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)
    return {"rewritten_query": response.content}

@tool
def retrieve_documents(rewritten_query: str):
    """Retrieves documents from the vector store."""
    response = embedder.call([rewritten_query], model_kwargs={"model": "thenlper/gte-base"})
    query_embedding = np.array([item.embedding for item in response.data])
    retrieved_documents = vector_store.search(query_embedding)
    return {"documents": retrieved_documents}

@tool
def graph_expander(documents: List[dict]):
    """Placeholder for graph expansion. Currently just passes documents through."""
    return {"expanded_documents": documents}

@tool
def rerank_documents(expanded_documents: List[dict], rewritten_query: str):
    """Reranks documents based on keyword matching with the query."""
    query_keywords = set(rewritten_query.lower().split())

    def score(doc):
        doc_keywords = set(doc['text'].lower().split())
        return len(query_keywords.intersection(doc_keywords))

    reranked = sorted(expanded_documents, key=score, reverse=True)
    return {"reranked_documents": reranked}

@tool
def generate_response(reranked_documents: List[dict], rewritten_query: str):
    """Generates a response using the LLM and cites the sources."""
    if not OPENAI_API_KEY:
        return {"generation": "No OPENAI_API_KEY found. Cannot generate a response."}

    context = "\n\n".join([doc["text"] for doc in reranked_documents])
    prompt = f"""
    Based on the following context, answer the user's query. Provide citations for each piece of information by referencing the document text.

    Context:
    {context}

    Query: {rewritten_query}

    Answer with citations:
    """
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)
    return {"generation": response.content}


# --- Graph Definition ---
workflow = StateGraph(AgentState)

# --- Nodes ---
workflow.add_node("rewrite_query", lambda state: {"rewritten_query": rewrite_query.invoke({"messages": state["messages"]})["rewritten_query"]})
workflow.add_node("retrieve_documents", lambda state: {"documents": retrieve_documents.invoke({"rewritten_query": state["rewritten_query"]})["documents"]})
workflow.add_node("graph_expander", lambda state: {"expanded_documents": graph_expander.invoke({"documents": state["documents"]})["expanded_documents"]})
workflow.add_node("rerank_documents", lambda state: {"reranked_documents": rerank_documents.invoke({"expanded_documents": state["expanded_documents"], "rewritten_query": state["rewritten_query"]})["reranked_documents"]})
workflow.add_node("generate_response", lambda state: {"generation": generate_response.invoke({"reranked_documents": state["reranked_documents"], "rewritten_query": state["rewritten_query"]})["generation"]})


# --- Edges ---
workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "graph_expander")
workflow.add_edge("graph_expander", "rerank_documents")
workflow.add_edge("rerank_documents", "generate_response")
workflow.add_edge("generate_response", END)

# --- Compile Graph ---
app = workflow.compile()

# --- Example Usage ---
if __name__ == "__main__":
    initialize_retriever()
    inputs = {"messages": [HumanMessage(content="Who is Dr. Liu?")]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")