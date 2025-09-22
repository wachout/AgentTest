import asyncio
import os
import sys
import re
import json
from pprint import pprint
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

# Import our custom modules
from graph_builder import create_graph, AgentState
from knowledge_extractor import create_knowledge_extractor
from dataclasses import asdict

# --- Configuration ---
LONG_TEXT_THRESHOLD = 20000
CHUNK_SIZE = 20000
OVERLAP_SIZE = 1000

# All helper functions remain the same as before...
def preprocess_html_blocks(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    html_spans = [m.span() for m in re.finditer(r'<html>.*?</html>', text, re.DOTALL | re.IGNORECASE)]
    if not html_spans: return text, []
    merged_spans = []
    if html_spans:
        current_start, current_end = html_spans[0]
        for next_start, next_end in html_spans[1:]:
            if re.fullmatch(r'\s*', text[current_end:next_start]): current_end = next_end
            else:
                merged_spans.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_spans.append((current_start, current_end))
    sanitized_text_parts, lookup_map, last_original_end, current_sanitized_pos = [], [], 0, 0
    for i, (start, end) in enumerate(merged_spans):
        placeholder = f"[--HTML_BLOCK_{i}--]"
        pre_html_text = text[last_original_end:start]
        sanitized_text_parts.append(pre_html_text)
        current_sanitized_pos += len(pre_html_text)
        sanitized_text_parts.append(placeholder)
        lookup_map.append({"placeholder": placeholder, "original_content_len": end - start, "sanitized_content_len": len(placeholder), "sanitized_start": current_sanitized_pos})
        current_sanitized_pos += len(placeholder)
        last_original_end = end
    sanitized_text_parts.append(text[last_original_end:])
    return "".join(sanitized_text_parts), lookup_map

def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
    if len(text) <= chunk_size: return [(text, 0)]
    chunks, offset = [], 0
    while offset < len(text):
        chunks.append((text[offset:offset + chunk_size], offset))
        offset += chunk_size - overlap
        if offset + overlap >= len(text): break
    return chunks

def convert_sanitized_indices_to_global(sanitized_indices: List[int], lookup_map: List[Dict[str, Any]]) -> List[int]:
    if not lookup_map: return sanitized_indices
    global_indices = []
    for sanitized_idx in sanitized_indices:
        offset = 0
        for block in lookup_map:
            if block['sanitized_start'] < sanitized_idx:
                offset += block['original_content_len'] - block['sanitized_content_len']
        global_indices.append(sanitized_idx + offset)
    return global_indices

def split_text_by_indices(text: str, indices: List[int]) -> List[str]:
    if not indices: return [text]
    segments, indices = [], sorted(list(set([0] + indices)))
    if len(text) not in indices: indices.append(len(text))
    for i in range(len(indices) - 1):
        segment = text[indices[i]:indices[i+1]]
        if segment: segments.append(segment.strip())
    return segments

async def analyze_chunk(graph_app, chunk_text: str) -> Dict[str, Any]:
    initial_state: AgentState = {"input_text": chunk_text, "chapter_splits": [], "paragraph_splits": [], "semantic_splits": [], "error": None}
    return await graph_app.ainvoke(initial_state)

async def main(filename: str = "sample_text.txt", llm_provider: str = "deepseek"):
    """Main execution function with knowledge extraction."""
    load_dotenv()

    print(f"Reading text from '{filename}'...")
    try:
        with open(filename, "r", encoding="utf-8") as f: original_text = f.read()
    except FileNotFoundError:
        print(f"ERROR: {filename} not found."); return

    print("Pre-processing text to handle HTML blocks...")
    sanitized_text, lookup_map = preprocess_html_blocks(original_text)

    graph_app = create_graph()
    print("Analyzing text for split points (chunking if necessary)...")
    chunks_with_offsets = chunk_text_with_overlap(sanitized_text, CHUNK_SIZE, OVERLAP_SIZE) if len(sanitized_text) > LONG_TEXT_THRESHOLD else [(sanitized_text, 0)]

    split_tasks = [analyze_chunk(graph_app, chunk) for chunk, offset in chunks_with_offsets]
    split_results = await asyncio.gather(*split_tasks)

    print("Aggregating and converting indices...")
    agg_sanitized_indices = {"chapters": set(), "paragraphs": set(), "semantics": set()}
    for i, result in enumerate(split_results):
        _, offset = chunks_with_offsets[i]
        for local_idx in result.get("chapter_splits", []): agg_sanitized_indices["chapters"].add(offset + local_idx)
        for local_idx in result.get("paragraph_splits", []): agg_sanitized_indices["paragraphs"].add(offset + local_idx)
        for local_idx in result.get("semantic_splits", []): agg_sanitized_indices["semantics"].add(offset + local_idx)

    final_global_indices = {key: convert_sanitized_indices_to_global(sorted(list(val)), lookup_map) for key, val in agg_sanitized_indices.items()}

    print("Generating final text segments...")
    final_segments = {key: split_text_by_indices(original_text, val) for key, val in final_global_indices.items()}

    print(f"Extracting knowledge from each text segment using '{llm_provider}'...")
    extractor = create_knowledge_extractor(llm_provider=llm_provider)
    final_output = {}

    for key, segments in final_segments.items():
        print(f"-> Processing {len(segments)} segments for '{key}'...")
        extraction_tasks = [extractor.extract(seg) for seg in segments]
        knowledge_graphs = await asyncio.gather(*extraction_tasks)

        final_output[key] = [{"segment_text": seg, "knowledge_graph": asdict(kg) if kg else None} for seg, kg in zip(segments, knowledge_graphs)]

    print("\n--- Final Structured Analysis Results ---")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    llm_provider = "deepseek"
    target_file = "sample_text.txt"

    args = sys.argv[1:]
    if args and not args[0].startswith('--'):
        target_file = args.pop(0)

    if "--llm" in args:
        try:
            llm_index = args.index("--llm")
            llm_provider = args[llm_index + 1]
            if llm_provider not in ["deepseek", "alibaba"]:
                print(f"Error: Invalid LLM provider '{llm_provider}'. Choose 'deepseek' or 'alibaba'.")
                sys.exit(1)
        except IndexError:
            print("Error: --llm flag requires an argument ('deepseek' or 'alibaba').")
            sys.exit(1)

    asyncio.run(main(filename=target_file, llm_provider=llm_provider))
