import asyncio
import os
import sys
from pprint import pprint
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

# Import the graph builder function from our other module
from graph_builder import create_graph, AgentState

# --- Configuration ---
LONG_TEXT_THRESHOLD = 20000
CHUNK_SIZE = 20000
OVERLAP_SIZE = 1000

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
    """
    Splits a long text into overlapping chunks.
    Returns a list of tuples, where each tuple contains the chunk of text and its starting offset.
    """
    if len(text) <= chunk_size:
        return [(text, 0)]

    chunks = []
    offset = 0
    while offset < len(text):
        chunk_end = offset + chunk_size
        chunks.append((text[offset:chunk_end], offset))

        # Move to the next chunk start position
        offset += chunk_size - overlap
        if offset + overlap >= len(text):
            break

    return chunks

async def analyze_chunk(graph_app, chunk_text: str) -> Dict[str, Any]:
    """Helper function to run analysis on a single chunk."""
    initial_state: AgentState = {
        "input_text": chunk_text,
        "chapter_splits": [],
        "paragraph_splits": [],
        "semantic_splits": [],
        "error": None,
    }
    return await graph_app.ainvoke(initial_state)


async def main(filename: str = "sample_text.txt"):
    """
    The main execution function for our multi-agent text splitting system.
    Handles both short and long texts with chunking logic.
    """
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        return

    print(f"Reading text from '{filename}'...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"ERROR: {filename} not found.")
        return

    graph_app = create_graph()

    if len(text_content) <= LONG_TEXT_THRESHOLD:
        print("Text is short. Analyzing as a single document.")
        chunks_with_offsets = [(text_content, 0)]
    else:
        print(f"Text is long ({len(text_content)} chars). Splitting into overlapping chunks.")
        chunks_with_offsets = chunk_text_with_overlap(text_content, CHUNK_SIZE, OVERLAP_SIZE)
        print(f"Created {len(chunks_with_offsets)} chunks to analyze.")

    # Create and run analysis tasks for all chunks in parallel
    tasks = [analyze_chunk(graph_app, chunk) for chunk, offset in chunks_with_offsets]
    print("\n--- Invoking Multi-Agent Graph on all chunks ---")
    results = await asyncio.gather(*tasks)
    print("\n--- Graph Execution Complete for all chunks ---")

    # --- Aggregate and process results ---
    all_chapter_splits = set()
    all_paragraph_splits = set()
    all_semantic_splits = set()

    for i, result in enumerate(results):
        chunk_text, offset = chunks_with_offsets[i]
        if result.get("error"):
            print(f"Warning: Chunk {i} failed with error: {result['error']}")
            continue

        # Convert local indices to global indices and add to sets (for auto-deduplication)
        for local_idx in result.get("chapter_splits", []):
            all_chapter_splits.add(offset + local_idx)
        for local_idx in result.get("paragraph_splits", []):
            all_paragraph_splits.add(offset + local_idx)
        for local_idx in result.get("semantic_splits", []):
            all_semantic_splits.add(offset + local_idx)

    # Sort the final lists
    final_results = {
        "chapter_splits": sorted(list(all_chapter_splits)),
        "paragraph_splits": sorted(list(all_paragraph_splits)),
        "semantic_splits": sorted(list(all_semantic_splits)),
    }

    print("\n--- Final Aggregated Analysis Results ---")
    pprint(final_results)


if __name__ == "__main__":
    # Allows passing a filename from the command line, e.g., `python main.py long_sample_text.txt`
    target_file = sys.argv[1] if len(sys.argv) > 1 else "sample_text.txt"
    asyncio.run(main(filename=target_file))
