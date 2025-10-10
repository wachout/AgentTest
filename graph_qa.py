import os
import asyncio
import shutil
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.base import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
import dashscope
from dashscope import Generation
from langchain_community.embeddings import DashScopeEmbeddings
import networkx as nx
from pyvis.network import Network

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
tongyi_api_key = os.getenv("TONGYI_API_KEY")
if not tongyi_api_key:
    raise ValueError("TONGYI_API_KEY not found in environment variables. Please create a .env file and add your TONGYI_API_KEY.")

# Set the Dashscope API key
dashscope.api_key = tongyi_api_key

# Initialize embeddings
embeddings = DashScopeEmbeddings(
    dashscope_api_key=tongyi_api_key,
    model="text-embedding-v1"
)

async def llm_wrapper(prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs):
    """
    Asynchronous wrapper for the Tongyi LLM using the dashscope library directly.
    Constructs a message list from the system prompt, history, and current prompt.
    """
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    if history_messages:
        for msg in history_messages:
            if msg.get("role") == "user":
                messages.append({'role': 'user', 'content': msg.get("content")})
            elif msg.get("role") == "assistant":
                messages.append({'role': 'assistant', 'content': msg.get("content")})

    messages.append({'role': 'user', 'content': prompt})

    # Make the API call using dashscope
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        result_format='message',
        temperature=0.7,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        print(f"LLM Error Code: {response.code}")
        print(f"LLM Error Message: {response.message}")
        return "{}"

async def embed_wrapper(texts: list[str], **kwargs):
    return embeddings.embed_documents(texts, **kwargs)

def verify_and_visualize_graph(graph_path="lightrag_data/graph_chunk_entity_relation.graphml"):
    """
    Reads the generated graphml file, prints its stats, and creates an HTML visualization.
    """
    print("\n--- Graph Verification and Visualization ---")
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return

    try:
        # Read the graph from the file
        G = nx.read_graphml(graph_path)

        # Print statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"Successfully loaded the knowledge graph.")
        print(f" - Number of nodes: {num_nodes}")
        print(f" - Number of edges: {num_edges}")

        if num_nodes == 0:
            print("Warning: The knowledge graph is empty. No visualization will be generated.")
            return

        # Create a Pyvis network for visualization
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
        net.from_nx(G)

        # Generate the visualization
        output_filename = "knowledge_graph.html"
        net.show(output_filename)
        print(f"Successfully generated graph visualization: {output_filename}")

    except Exception as e:
        print(f"An error occurred during graph verification: {e}")


async def main():
    """
    Main function to run the RAG pipeline.
    Initializes LightRAG, ingests documents, performs a query, and verifies the graph.
    """
    # Although the user requested "synchronous" calls, the lightrag library is
    # async-native. Using asyncio.run() ensures that the async operations
    # complete sequentially, which achieves the user's goal of preventing
    # file write failures due to race conditions.

    # 1. Initialize LightRAG
    rag = LightRAG(
        llm_model_func=llm_wrapper,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            func=embed_wrapper
        ),
        working_dir="./lightrag_data"
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # 2. Data Ingestion
    try:
        with open("kb1.txt", "r", encoding="utf-8") as f:
            kb1_content = f.read()
        with open("kb2.txt", "r", encoding="utf-8") as f:
            kb2_content = f.read()
    except FileNotFoundError as e:
        print(f"Error: Knowledge base file not found. {e}")
        print("Please ensure kb1.txt and kb2.txt are in the same directory.")
        return

    await asyncio.gather(
        rag.ainsert(kb1_content, ids=["kb1"]),
        rag.ainsert(kb2_content, ids=["kb2"])
    )
    print("Successfully inserted knowledge base documents.")

    # 3. Query
    query = "流浪地球小说和电影有什么区别？"
    print(f"\nQuerying with: '{query}'")

    response = await rag.aquery(query, param=QueryParam(mode="mix"))

    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    # Per the user's request, clear the cache directory to ensure a fresh run
    # and prevent file writing failures.
    if os.path.exists("./lightrag_data"):
        shutil.rmtree("./lightrag_data")
        print("Cleared previous lightrag_data directory.")

    asyncio.run(main())

    # 4. Verify and Visualize the Knowledge Graph
    # This is a synchronous step after the main async pipeline has completed.
    verify_and_visualize_graph()