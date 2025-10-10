import os
import json
from dataclasses import asdict
from typing import List, Type

# LightRAG imports for data structure and parsing
from lightrag.core.base_data_class import DataClass
from lightrag.components.output_parsers import JsonOutputParser
from dataclasses import dataclass, field

# LangChain import for type hinting the LLM object
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# python-dotenv for loading environment variables
from dotenv import load_dotenv

# Note on the implementation:
# This solution follows the "hybrid" approach discussed. It uses `lightrag` for what it excels at
# in this context: defining a structured output (`KnowledgeGraph`) and parsing the LLM's raw
# string output into that structure (`JsonOutputParser`).
# It bypasses the complex `lightrag.Generator` and `lightrag.ModelClient` and instead uses the
# user-provided LangChain LLM object directly (`llm.invoke`), as requested in the code review.
# This results in a clean, robust, and reusable function.

# 1. Define the data structure for the output using LightRAG's DataClass
# This ensures compatibility with LightRAG's parsers.
@dataclass
class Node(DataClass):
    """Represents a single node in the knowledge graph."""
    id: str = field(metadata={"description": "A unique identifier for the entity (e.g., 'John_Doe')."})
    type: str = field(metadata={"description": "The category of the entity (e.g., 'Person', 'Organization')."})

@dataclass
class Edge(DataClass):
    """Represents a single edge (relationship) in the knowledge graph."""
    source: str = field(metadata={"description": "The 'id' of the source node."})
    target: str = field(metadata={"description": "The 'id' of the target node."})
    label: str = field(metadata={"description": "A brief description of the relationship (e.g., 'works_at')."})

@dataclass
class KnowledgeGraph(DataClass):
    """Represents the entire knowledge graph with a list of nodes and edges."""
    nodes: List[Node]
    edges: List[Edge]

# 2. Define the core, reusable graph extraction function
def extract_graph(text: str, llm: BaseChatModel) -> KnowledgeGraph:
    """
    Extracts a knowledge graph from a given text using a specified LangChain LLM.

    Args:
        text: The input text to process.
        llm: An instantiated LangChain chat model (e.g., ChatOpenAI, ChatTongyi).

    Returns:
        A KnowledgeGraph object containing the extracted nodes and edges.
    """
    # Use LightRAG's parser to validate and structure the final output
    output_parser = JsonOutputParser(data_class=KnowledgeGraph, return_data_class=True)

    # A simpler, more direct prompt with a clear example is more reliable.
    prompt_template = f"""Your goal is to extract a knowledge graph from the provided text.
Format the output as a single JSON object with "nodes" and "edges" keys.

EXAMPLE FORMAT:
```json
{{
  "nodes": [
    {{"id": "Person_Name", "type": "Person"}},
    {{"id": "Company_Name", "type": "Organization"}}
  ],
  "edges": [
    {{"source": "Person_Name", "target": "Company_Name", "label": "works_at"}}
  ]
}}
```

TEXT TO PROCESS:
---
{text}
---

Provide only the valid JSON object as your response.
"""

    print("--- Sending Prompt to LLM ---")
    print(prompt_template)
    print("-----------------------------\n")

    # Directly invoke the user-provided LLM with the complete prompt
    response = llm.invoke(prompt_template)
    response_content = response.content

    print("--- Received Raw Response from LLM ---")
    print(response_content)
    print("--------------------------------------\n")

    # Use the LightRAG parser to parse the raw string response into our data class
    try:
        # The parser expects a string, so we pass the content of the AIMessage
        parsed_graph = output_parser.call(response_content)
        return parsed_graph
    except Exception as e:
        print(f"Error parsing the LLM response: {e}")
        print("Returning an empty graph.")
        return KnowledgeGraph(nodes=[], edges=[])

# 3. Provide an example of how to use the function
if __name__ == "__main__":
    # Load API keys from a .env file for security
    # Create a file named .env in the same directory with your keys:
    # DEEPSEEK_API_KEY="sk-..."
    # DEEPSEEK_BASE_URL="..."
    load_dotenv()

    # Get credentials from environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment variables.")
        print("Please create a .env file and add your key.")
    else:
        # Instantiate the LLM using the user's preferred method
        llm = ChatOpenAI(
            temperature=0.6,
            model="deepseek-reasoner",
            api_key=api_key,
            base_url=base_url,
        )

        # Define the sample text for extraction
        sample_text = """
        John Doe, a software engineer at Innovate Inc., is working on a new project called "Orion".
        Maria Garcia, who is the project manager for Orion, is based in the New York office.
        Innovate Inc. was founded in 2015 and has its headquarters in San Francisco.
        """

        print("Starting graph extraction process...")
        # Call the reusable function with the text and the LLM
        knowledge_graph_result = extract_graph(text=sample_text, llm=llm)

        # Print the final, structured result
        if knowledge_graph_result and (knowledge_graph_result.nodes or knowledge_graph_result.edges):
            print("--- Final Extracted Graph Data ---")
            # Use asdict to convert the dataclass object to a dictionary for pretty printing
            print(json.dumps(asdict(knowledge_graph_result), indent=2))
            print("----------------------------------\n")
            print("Graph extraction successful.")
        else:
            print("Graph extraction failed or returned an empty graph.")