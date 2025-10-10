import os
import argparse
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# --- 1. Get API Key from Command-Line Arguments ---
# This approach is robust and avoids file system issues encountered in this environment.
parser = argparse.ArgumentParser(description="Extract a knowledge graph from text using an API key.")
parser.add_argument(
    "--api-key",
    type=str,
    required=True,
    help="The DeepSeek API key."
)
args = parser.parse_args()
DEEPSEEK_API_KEY = args.api_key

# --- 2. Define Data Structures for the Graph (using Pydantic V2) ---
# These models ensure the LLM provides a structured, predictable output.
class Node(BaseModel):
    """Represents a single entity or concept in the knowledge graph."""
    id: str = Field(..., description="A unique identifier for the node (e.g., 'John Doe').")
    type: str = Field(..., description="The type or category of the node (e.g., 'Person', 'Company').")

class Edge(BaseModel):
    """Represents a relationship between two nodes in the knowledge graph."""
    source: str = Field(..., description="The ID of the source node.")
    target: str = Field(..., description="The ID of the target node.")
    label: str = Field(..., description="The description of the relationship (e.g., 'is CEO of').")

class Graph(BaseModel):
    """Represents the entire knowledge graph extracted from the text."""
    nodes: List[Node] = Field(..., description="A list of all nodes in the graph.")
    edges: List[Edge] = Field(..., description="A list of all edges connecting the nodes.")

# --- 3. Configure the Language Model ---
# We use LangChain's ChatOpenAI, which provides a stable interface to OpenAI-compatible APIs.
llm = ChatOpenAI(
    temperature=0.0,
    model="deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# --- 4. Create the Extractor Chain ---
# This chain defines the prompt and instructs the LLM to return structured output.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at extracting knowledge graphs from text. "
            "Your task is to identify all the entities and their relationships. "
            "Return the extracted data as a single JSON object that strictly follows the provided schema.",
        ),
        ("human", "Extract the knowledge graph from the following text:\n\n{text_input}"),
    ]
)

# We explicitly use `method="function_calling"` as it is more broadly supported
# by models for structured output than the default "json_mode".
extractor = prompt | llm.with_structured_output(Graph, method="function_calling")

# --- 5. Main Execution Block ---
def main():
    """Main function to run the graph extraction process."""
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: data.txt not found. Please create it with the text to analyze.")
        return

    print("--- Input Text ---")
    print(text)
    print("\n" + "="*50 + "\n")

    # Run the extractor by invoking it with the input text
    try:
        graph_data = extractor.invoke({"text_input": text})

        if graph_data:
            print("--- Extracted Knowledge Graph (using langchain) ---")
            print(graph_data.model_dump_json(indent=2))
        else:
            print("Could not extract a valid graph from the text.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()