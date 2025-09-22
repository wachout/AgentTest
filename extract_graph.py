import os
import asyncio
import json
import networkx as nx
import numpy as np
import shutil
import sys
from typing import Dict, Any, List

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
OUTPUT_JSON_PATH = os.path.join(WORKING_DIR, "output_graph.json")

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
async def extract_and_save_graph(text_segments: List[str]) -> str:
    """
    Performs graph extraction from a list of text segments, saves the result
    to a JSON file, and returns the absolute path to that file.
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

    # Combine text segments and process
    full_text = "\n".join(text_segments)
    print(f"Processing {len(text_segments)} text segment(s)...", file=sys.stderr)
    await rag.ainsert([full_text], ids=["input_text"])
    await rag.finalize_storages()

    print(f"Reading persisted graph from {GRAPH_FILE_PATH}...", file=sys.stderr)
    if not os.path.exists(GRAPH_FILE_PATH):
        raise FileNotFoundError(f"LightRAG did not create the expected graph file at {GRAPH_FILE_PATH}")

    graph = nx.read_graphml(GRAPH_FILE_PATH)

    # Relabel nodes to use entity names as IDs
    id_to_name_mapping = {node_id: data.get('id', node_id) for node_id, data in graph.nodes(data=True)}
    relabelled_graph = nx.relabel_nodes(graph, id_to_name_mapping, copy=True)
    graph_json_data = nx.node_link_data(relabelled_graph)

    # Save the final JSON to a file
    print(f"Saving extracted graph to {OUTPUT_JSON_PATH}...", file=sys.stderr)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_json_data, f, ensure_ascii=False, indent=4)

    # Verify that the file was created
    if not os.path.exists(OUTPUT_JSON_PATH):
        raise IOError(f"Failed to save the output JSON file to {OUTPUT_JSON_PATH}")

    abs_path = os.path.abspath(OUTPUT_JSON_PATH)
    print(f"File saved successfully to {abs_path}", file=sys.stderr)

    return abs_path

def notify_completion(file_path: str):
    """
    Prints a user-friendly completion message.
    """
    if file_path and os.path.exists(file_path):
        message = f"\n✅ 图谱抽取已成功完成！\n"
        message += f"   - 结果已保存到文件: {file_path}"
        print(message, file=sys.stderr)
    else:
        print("\n❌ 抽取失败或文件未保存。", file=sys.stderr)

async def main():
    """
    Main execution block to demonstrate the new workflow.
    """
    setup_logger("lightrag", level="INFO")
    print("--- Starting Graph Extraction Process ---", file=sys.stderr)

    try:
        # 1. Define input text segments
        input_texts = [
            "《流浪地球》是刘慈欣创作的科幻小说。",
            "故事的核心是太阳即将氦闪，毁灭太阳系。为了生存，人类联合政府启动了“流浪地球”计划，在地球表面建造了一万座巨大的行星发动机，推动地球离开太阳系，前往半人马座比邻星，开启一场长达2500年的星际流浪。",
            "主角刘启是一名年轻的航天员，他的父亲刘培强是国际空间站的宇航员。他们在木星引力危机中，为了拯救地球，做出了巨大的牺牲。"
        ]

        # 2. Call the function to perform extraction and get the file path
        output_file_path = await extract_and_save_graph(input_texts)

        # 3. Call the notification function
        notify_completion(output_file_path)

        # 4. Demonstrate using the returned path to read the data
        print(f"\n--- Reading data from returned path: {output_file_path} ---", file=sys.stderr)
        with open(output_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(json.dumps(data, ensure_ascii=False, indent=4))


    except Exception as e:
        print(f"\nAn error occurred during the process: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
