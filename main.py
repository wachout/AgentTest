import os
import sys
import json
import re
from typing import List, TypedDict
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    if len(document_text) > 20000:
        print("--- Document is long, using recursive character splitter. ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000, length_function=len)
        chunks = text_splitter.split_text(document_text)
        return [Document(page_content=chunk) for chunk in chunks]
    else:
        print("--- Document is short, using headings for splitting. ---")
        if not headings:
            return [Document(page_content=document_text)]

        headings.sort(key=len, reverse=True)
        pattern = '|'.join(map(re.escape, headings))
        chunks = re.split(f'({pattern})', document_text)

        documents = []
        if chunks[0]:
            documents.append(Document(page_content=chunks[0].strip()))

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
    verification_status: str

def split_node(state: GraphState):
    """
    Node to parse the TOC and split the document text into chunks.
    """
    print("--- Splitting Document... ---")
    document_text = state["document_text"]
    toc_data = state["toc_data"]

    headings = parse_json_toc(toc_data)
    chunks = split_text_by_headings(document_text, headings)
    return {"chunks": chunks}

def verify_integrity_node(state: GraphState):
    """
    Node to verify the integrity of the split by reassembling the chunks.
    """
    print("--- Verifying Splitting Integrity... ---")
    original_text = state["document_text"]
    chunks = state["chunks"]

    # Reassemble the text from chunks
    reassembled_text = "".join(doc.page_content for doc in chunks)

    # Normalize both texts by removing all whitespace for a robust comparison
    original_normalized = re.sub(r'\s+', '', original_text)
    reassembled_normalized = re.sub(r'\s+', '', reassembled_text)

    if original_normalized == reassembled_normalized:
        status = "Success: Reassembled text matches original text."
    else:
        status = "Failure: Reassembled text does not match original text."
        # Optional: Print lengths for debugging
        # print(f"Original length (normalized): {len(original_normalized)}")
        # print(f"Reassembled length (normalized): {len(reassembled_normalized)}")

    return {"verification_status": status}

def main():
    """
    Main function to set up and run the text splitting workflow.
    """
    # 1. Define the workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("split", split_node)
    workflow.add_node("verify", verify_integrity_node)

    workflow.add_edge(START, "split")
    workflow.add_edge("split", "verify")
    workflow.add_edge("verify", END)
    app = workflow.compile()

    # 2. Prepare the input data from files
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
    }

    # 3. Run the workflow
    final_state = app.invoke(inputs)

    # 4. Print the final output
    print("\n--- Workflow Complete. Final Chunks: ---")
    for i, chunk in enumerate(final_state['chunks']):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
    print("\n-----------------------------------------")
    print(f"Total chunks created: {len(final_state['chunks'])}")
    print(f"Integrity Check: {final_state['verification_status']}")


if __name__ == "__main__":
    main()
