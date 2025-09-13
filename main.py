import os
import sys
import json
from typing import List, TypedDict, Optional
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

def parse_json_toc(toc_data: dict) -> List[str]:
    """
    Recursively parses a JSON/dict table of contents to extract all headings.
    """
    headings = []
    for key, value in toc_data.items():
        headings.append(key)
        if isinstance(value, dict) and value:
            headings.extend(parse_json_toc(value))
    return headings

def split_text_by_headings(document_text: str, headings: List[str]) -> List[Document]:
    """
    Splits the document text based on a list of headings or into overlapping chunks if it's too long.
    """
    # For documents over 20,000 characters, use a recursive splitter with overlap.
    if len(document_text) > 20000:
        print("Document is long, using recursive character splitter.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000, length_function=len)
        chunks = text_splitter.split_text(document_text)
        return [Document(page_content=chunk) for chunk in chunks]
    # For shorter documents, split based on the provided headings.
    else:
        print("Document is short, using headings for splitting.")
        if not headings:
            return [Document(page_content=document_text)]

        # Create a regex pattern to split the document by the headings.
        # The headings are sorted by length descending to handle nested headings correctly.
        headings.sort(key=len, reverse=True)
        pattern = '|'.join(map(re.escape, headings))
        chunks = re.split(f'({pattern})', document_text)

        documents = []
        # The first chunk is any text before the first heading.
        if chunks[0]:
            documents.append(Document(page_content=chunks[0].strip()))

        # Combine each heading with the text that follows it.
        for i in range(1, len(chunks), 2):
            heading = chunks[i]
            text = chunks[i+1] if (i+1) < len(chunks) else ""
            documents.append(Document(page_content=(heading + text).strip()))

        return [doc for doc in documents if doc.page_content]

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    document_text: str
    toc_data: dict
    chunks: List[Document]
    vector_store: Optional[FAISS]
    llm: ChatOpenAI
    embeddings: Embeddings
    question: str
    answer: str

def split_node(state: GraphState):
    """
    Node to parse the TOC and split the document text into chunks.
    """
    print("--- Splitting Document ---")
    document_text = state["document_text"]
    toc_data = state["toc_data"]

    headings = parse_json_toc(toc_data)
    chunks = split_text_by_headings(document_text, headings)
    return {"chunks": chunks}

def index_node(state: GraphState):
    """
    Node to create a FAISS vector store from the text chunks.
    """
    print("--- Indexing Chunks ---")
    chunks = state["chunks"]
    embeddings = state["embeddings"]
    vector_store = FAISS.from_documents(chunks, embeddings)
    return {"vector_store": vector_store}

def rag_node(state: GraphState):
    """
    Node to perform Retrieval-Augmented Generation.
    """
    print("--- Retrieving and Generating Answer ---")
    question = state["question"]
    vector_store = state["vector_store"]
    llm = state["llm"]

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = prompt | llm
    answer = rag_chain.invoke({"question": question, "context": context})
    return {"answer": answer.content}

def main():
    """
    Main function to set up and run the agentic workflow.
    """
    load_dotenv()

    llm = ChatOpenAI(
        temperature=0.6,
        model="deepseek-reasoner",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    # Using a multilingual model better suited for Chinese text
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    workflow = StateGraph(GraphState)
    workflow.add_node("split", split_node)
    workflow.add_node("index", index_node)
    workflow.add_node("rag", rag_node)

    workflow.add_edge(START, "split")
    workflow.add_edge("split", "index")
    workflow.add_edge("index", "rag")
    workflow.add_edge("rag", END)
    app = workflow.compile()

    try:
        with open("document.txt", "r", encoding="utf-8") as f:
            document_text = f.read()
        with open("toc.json", "r", encoding="utf-8") as f:
            toc_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'document.txt' and 'toc.json' are in the same directory.")
        sys.exit(1)

    inputs = {
        "document_text": document_text,
        "toc_data": toc_data,
        "llm": llm,
        "embeddings": embeddings,
        "question": "项目管理单位的名称是什么?"
    }

    final_state = app.invoke(inputs)

    print("\n--- Workflow Complete ---")
    print(f"Question: {final_state['question']}")
    print(f"Answer: {final_state['answer']}")

if __name__ == "__main__":
    main()
