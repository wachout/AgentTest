import os
import asyncio
import json
import networkx as nx
import numpy as np
import shutil
import sys
from typing import Dict, Any

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# 1. API Configuration
API_KEY = "sk-0270be722a48439e9ed73001e8e2524b"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "qwen-max"
EMBEDDING_MODEL = "text-embedding-v2"
WORKING_DIR = "./lightrag_data"
GRAPH_FILE_PATH = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")

# 2. Custom Model Handlers
async def qwen_llm_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
    model_to_use = kwargs.pop('model', CHAT_MODEL)
    kwargs.pop('context', None)
    return await openai_complete_if_cache(
        prompt=prompt, model=model_to_use, api_key=API_KEY, base_url=BASE_URL,
        system_prompt=system_prompt, **kwargs
    )

async def qwen_embedding_func(texts: list[str], **kwargs) -> list[list[float]]:
    response_array = await openai_embed(
        texts=texts, model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL, **kwargs
    )
    if isinstance(response_array, np.ndarray):
        return response_array.tolist()
    elif hasattr(response_array, 'data'):
        return [item.embedding for item in response_array.data]
    raise TypeError(f"Unexpected response type from embedding function: {type(response_array)}")

# 3. Main Application Logic
async def extract_graph_from_text(text_content: str) -> Dict[str, Any]:
    """
    Performs graph extraction from a given text and returns the graph as a JSON object.
    This function encapsulates the entire LightRAG process and works around a library
    bug by reading the persisted graph file instead of accessing the in-memory object.
    """
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    print("Initializing LightRAG...", file=sys.stderr)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=qwen_llm_func,
        embedding_func=EmbeddingFunc(func=qwen_embedding_func, embedding_dim=1536),
        graph_storage="NetworkXStorage",
        addon_params={"language": "Simplified Chinese"}
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    print(f"Processing text...", file=sys.stderr)
    await rag.ainsert([text_content], ids=["input_text"])
    await rag.finalize_storages()

    print(f"Reading persisted graph from {GRAPH_FILE_PATH}...", file=sys.stderr)
    if not os.path.exists(GRAPH_FILE_PATH):
        raise FileNotFoundError(f"LightRAG did not create the expected graph file at {GRAPH_FILE_PATH}")

    graph = nx.read_graphml(GRAPH_FILE_PATH)

    # The node IDs in the graphml are hashes. We need to map them back to the entity names.
    # The entity names are stored as attributes on the nodes.
    # We will create a new graph with the entity names as IDs.

    # Create a mapping from old ID to new ID (entity name)
    id_to_name_mapping = {node_id: data.get('id', node_id) for node_id, data in graph.nodes(data=True)}

    # Create a new graph with renamed nodes
    relabelled_graph = nx.relabel_nodes(graph, id_to_name_mapping, copy=True)

    return nx.node_link_data(relabelled_graph)

def notify_completion(result: Dict[str, Any]):
    """
    Prints a user-friendly completion message and a summary of the results.
    """
    if result and result.get('nodes'):
        message = f"\n✅ 图谱抽取已成功完成！\n"
        message += f"   - 提取到 {len(result['nodes'])} 个实体 (Nodes)\n"
        message += f"   - 提取到 {len(result['links'])} 条关系 (Links)"
        print(message, file=sys.stderr)
    else:
        print("\n❌ 抽取失败或未抽取出任何内容。", file=sys.stderr)

async def main():
    """
    Main execution block to run the extraction and notification.
    """
    setup_logger("lightrag", level="INFO")
    print("--- Starting Graph Extraction Process ---", file=sys.stderr)

    try:
        with open("wandering_earth_sample.txt", "r", encoding="utf-8") as f:
            text_to_process = f.read()

        # 1. Extract the graph and get the JSON result
        graph_json = await extract_graph_from_text(text_to_process)

        # 2. Print the JSON result to stdout for consumption by other tools
        print(json.dumps(graph_json, ensure_ascii=False, indent=4))

        # 3. Call the notification function to inform the user
        notify_completion(graph_json)

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
