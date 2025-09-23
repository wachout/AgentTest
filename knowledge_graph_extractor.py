import os
import asyncio
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Basic configuration for logging
setup_logger("lightrag", level="INFO")
log = logging.getLogger("lightrag")

# --- Configuration ---
# Load credentials from environment variables for security
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = "deepseek-reasoner"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
WORKING_DIR = "./rag_storage"
INPUT_TEXT_FILE = "input.txt"

# --- Custom LLM and Embedding Functions ---

async def deepseek_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Custom LLM function for DeepSeek."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
    return await openai_complete_if_cache(
        DEEPSEEK_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        **kwargs
    )

# Cache the sentence transformer model so it's not reloaded on every call
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        log.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

async def local_embedding_func(texts: list[str]) -> np.ndarray:
    """Async custom embedding function using a local sentence-transformer model."""
    model = get_embedding_model()
    # Run the synchronous encode method in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(model.encode, texts, convert_to_numpy=True)


# --- Core Functions ---

async def initialize_rag():
    """Initializes the LightRAG instance with DeepSeek and local embeddings."""
    if not DEEPSEEK_API_KEY:
        log.error("DEEPSEEK_API_KEY environment variable not set.")
        raise ValueError("Please set your DEEPSEEK_API_KEY.")

    # Clear the working directory for a fresh start
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)


    # Get the embedding dimension from the model's config
    model = get_embedding_model()
    embedding_dim = model.get_sentence_embedding_dimension()
    log.info(f"Embedding dimension: {embedding_dim}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            func=local_embedding_func
        ),
        llm_model_func=deepseek_llm_func,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def process_text_file(rag: LightRAG, filepath: str):
    """Reads a text file and inserts its content into LightRAG."""
    log.info(f"Processing text file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if text:
            await rag.ainsert(text, ids=[filepath])
            # The insertion process runs in the background. We need to wait for it to complete.
            while True:
                # Use the synchronous get_doc_status method
                status = rag.get_doc_status(filepath)
                if status and status.get('status') == 'completed':
                    log.info(f"Document '{filepath}' processed successfully.")
                    break
                elif status and status.get('status') == 'failed':
                    log.error(f"Document '{filepath}' processing failed.")
                    break
                log.info(f"Waiting for document processing... Current status: {status.get('status') if status else 'Not found'}")
                await asyncio.sleep(2)
            log.info("Text successfully inserted and processed into LightRAG.")
        else:
            log.warning(f"The file {filepath} is empty.")
    except FileNotFoundError:
        log.error(f"Error: The file {filepath} was not found.")
        raise

async def get_all_entities(rag: LightRAG):
    """Retrieves and displays all entities from the graph."""
    log.info("Retrieving all entities...")
    if hasattr(rag.graph_storage, 'graph') and hasattr(rag.graph_storage.graph, 'nodes'):
        entities = rag.graph_storage.graph.nodes(data=True)
        print("\n--- All Entities (Nodes) ---")
        if entities:
            for entity, attributes in entities:
                print(f"  - Entity: {entity}")
                for key, value in attributes.items():
                    print(f"    - {key}: {value}")
        else:
            print("No entities found.")
    else:
        print("Could not retrieve entities. The graph storage does not seem to be a NetworkX graph or is empty.")

async def get_all_relations(rag: LightRAG):
    """Retrieves and displays all relations from the graph."""
    log.info("Retrieving all relations...")
    if hasattr(rag.graph_storage, 'graph') and hasattr(rag.graph_storage.graph, 'edges'):
        relations = rag.graph_storage.graph.edges(data=True)
        print("\n--- All Relations (Edges) ---")
        if relations:
            for source, target, attributes in relations:
                print(f"  - Relation from '{source}' to '{target}'")
                for key, value in attributes.items():
                    print(f"    - {key}: {value}")
        else:
            print("No relations found.")
    else:
        print("Could not retrieve relations. The graph storage does not seem to be a NetworkX graph or is empty.")

async def main():
    """Main function to run the knowledge graph extraction process."""
    rag = None
    try:
        rag = await initialize_rag()
        await process_text_file(rag, INPUT_TEXT_FILE)

        # 1. Get and display all entities (and their attributes)
        await get_all_entities(rag)

        # 2. Get and display all relations
        await get_all_relations(rag)

        # 3. Keywords are represented by the entities themselves.
        print("\n--- Keywords ---")
        print("Keywords are represented by the extracted entities.")

        # 4. Show how to export the data for graph database storage
        export_path = "knowledge_graph.csv"
        log.info(f"Exporting graph data to {export_path}...")
        await rag.aexport_data(export_path, file_format="csv")
        log.info(f"Data successfully exported to {export_path}.")
        print(f"\nGraph data has been exported to '{export_path}'. This file can be imported into a graph database like Neo4j.")
        print("To use Neo4j directly, you would initialize LightRAG like this:")
        print("rag = LightRAG(..., graph_storage=\"Neo4JStorage\")")
        print("And set the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")

    except Exception as e:
        log.error(f"An error occurred in the main process: {e}", exc_info=True)
    finally:
        if rag:
            await rag.finalize_storages()
            log.info("RAG storages finalized.")

if __name__ == "__main__":
    # Add a check for the API key before running
    if not DEEPSEEK_API_KEY:
        print("Error: The DEEPSEEK_API_KEY environment variable must be set.")
        print("Please set it before running the script, e.g.:")
        print("export DEEPSEEK_API_KEY='your_key_here'")
    else:
        asyncio.run(main())
