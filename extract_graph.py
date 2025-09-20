import os
import asyncio
import json
import networkx as nx
import numpy as np
import shutil
import sys
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# 1. Configure API keys and model details from user input
API_KEY = "sk-0270be722a48439e9ed73001e8e2524b"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-max"
EMBEDDING_MODEL = "text-embedding-v2"

# 2. Define custom functions for LightRAG to use the specified model
async def qwen_llm_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
    """
    Custom LLM function to call the Tongyi Qwen model.
    """
    model_to_use = kwargs.pop('model', CHAT_MODEL)
    kwargs.pop('context', None) # Remove context if it exists

    return await openai_complete_if_cache(
        prompt=prompt,
        model=model_to_use,
        api_key=API_KEY,
        base_url=BASE_URL,
        system_prompt=system_prompt,
        **kwargs,
    )

async def qwen_embedding_func(texts: list[str], **kwargs) -> list[list[float]]:
    """
    Custom embedding function to call the Tongyi embedding model.
    """
    response_array = await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )
    if isinstance(response_array, np.ndarray):
        return response_array.tolist()
    elif hasattr(response_array, 'data'):
        return [item.embedding for item in response_array.data]
    else:
        raise TypeError(f"Unexpected response type from embedding function: {type(response_array)}")

# 3. Main logic in an async function
async def main():
    """
    Main function to run the graph extraction process.
    """
    setup_logger("lightrag", level="INFO")

    working_dir = "./lightrag_data"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir)

    print("Initializing LightRAG with custom Tongyi Qwen models...", file=sys.stderr)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=qwen_llm_func,
        embedding_func=EmbeddingFunc(
            func=qwen_embedding_func,
            embedding_dim=1536
        ),
        addon_params={"language": "Simplified Chinese"}
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    print("Reading the sample text file...", file=sys.stderr)
    with open("wandering_earth_sample.txt", "r", encoding="utf-8") as f:
        text_content = f.read()

    print("Inserting text into LightRAG for processing...", file=sys.stderr)
    await rag.ainsert([text_content], ids=["wandering_earth_sample"])

    print("Extraction complete. Accessing the graph...", file=sys.stderr)
    graph = rag.graph_storage.graph

    if graph:
        print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.", file=sys.stderr)
        graph_json_data = nx.node_link_data(graph)

        # Instead of writing to a file, print the JSON to stdout
        print("\n--- EXTRACTED GRAPH DATA (JSON) ---", file=sys.stderr)
        print(json.dumps(graph_json_data, ensure_ascii=False, indent=4))
        print("\n--- END OF GRAPH DATA ---", file=sys.stderr)

    else:
        print("Graph could not be created.", file=sys.stderr)

    await rag.finalize_storages()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
