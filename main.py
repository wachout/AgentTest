import asyncio
import os
import sys
import re
from pprint import pprint
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any

# Import the graph builder function from our other module
from graph_builder import create_graph, AgentState

# --- Configuration ---
LONG_TEXT_THRESHOLD = 20000
CHUNK_SIZE = 20000
OVERLAP_SIZE = 1000

def preprocess_html_blocks(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finds, merges, and replaces HTML blocks with placeholders.
    The lookup map now includes sanitized text positions for easier post-processing.
    """
    html_spans = [m.span() for m in re.finditer(r'<html>.*?</html>', text, re.DOTALL | re.IGNORECASE)]
    if not html_spans:
        return text, []

    merged_spans = []
    if html_spans:
        current_start, current_end = html_spans[0]
        for next_start, next_end in html_spans[1:]:
            if re.fullmatch(r'\s*', text[current_end:next_start]):
                current_end = next_end
            else:
                merged_spans.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_spans.append((current_start, current_end))

    sanitized_text_parts = []
    lookup_map = []
    last_original_end = 0
    current_sanitized_pos = 0
    for i, (start, end) in enumerate(merged_spans):
        placeholder = f"[--HTML_BLOCK_{i}--]"

        # Add the text part before the HTML block
        pre_html_text = text[last_original_end:start]
        sanitized_text_parts.append(pre_html_text)
        current_sanitized_pos += len(pre_html_text)

        # Add the placeholder
        sanitized_text_parts.append(placeholder)

        lookup_map.append({
            "placeholder": placeholder,
            "original_content_len": end - start,
            "sanitized_content_len": len(placeholder),
            "sanitized_start": current_sanitized_pos,
        })

        current_sanitized_pos += len(placeholder)
        last_original_end = end

    sanitized_text_parts.append(text[last_original_end:])
    return "".join(sanitized_text_parts), lookup_map


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

def convert_sanitized_indices_to_global(
    sanitized_indices: List[int],
    lookup_map: List[Dict[str, Any]]
) -> List[int]:
    """Converts indices from the sanitized text back to global original-text indices."""
    if not lookup_map:
        return sanitized_indices

    global_indices = []
    for sanitized_idx in sanitized_indices:
        offset = 0
        for block in lookup_map:
            if block['sanitized_start'] < sanitized_idx:
                offset += block['original_content_len'] - block['sanitized_content_len']
        global_indices.append(sanitized_idx + offset)

    return global_indices

def split_text_by_indices(text: str, indices: List[int]) -> List[str]:
    """Splits a text into segments based on a list of starting indices."""
    if not indices:
        return [text]

    segments = []
    indices = sorted(list(set([0] + indices)))
    if len(text) not in indices:
        indices.append(len(text))

    for i in range(len(indices) - 1):
        start_idx, end_idx = indices[i], indices[i+1]
        segment = text[start_idx:end_idx]
        if segment:
            segments.append(segment.strip())

    return segments

async def analyze_chunk(graph_app, chunk_text: str) -> Dict[str, Any]:
    """Helper function to run analysis on a single chunk."""
    initial_state: AgentState = {"input_text": chunk_text, "chapter_splits": [], "paragraph_splits": [], "semantic_splits": [], "error": None}
    return await graph_app.ainvoke(initial_state)

async def main(filename: str = "sample_text.txt"):
    """Main execution function with pre-processing and post-processing for HTML."""
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        return

    print(f"Reading text from '{filename}'...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            original_text = f.read()
    except FileNotFoundError:
        print(f"ERROR: {filename} not found.")
        return

    print("Step 1: Pre-processing text to handle HTML blocks...")
    sanitized_text, lookup_map = preprocess_html_blocks(original_text)

    graph_app = create_graph()

    print("Step 2: Analyzing text (chunking if necessary)...")
    if len(sanitized_text) <= LONG_TEXT_THRESHOLD:
        chunks_with_offsets = [(sanitized_text, 0)]
    else:
        chunks_with_offsets = chunk_text_with_overlap(sanitized_text, CHUNK_SIZE, OVERLAP_SIZE)

    tasks = [analyze_chunk(graph_app, chunk) for chunk, offset in chunks_with_offsets]
    results = await asyncio.gather(*tasks)

    print("Step 3: Aggregating and converting indices...")
    agg_sanitized_indices = {"chapters": set(), "paragraphs": set(), "semantics": set()}
    for i, result in enumerate(results):
        _, offset = chunks_with_offsets[i]
        for local_idx in result.get("chapter_splits", []): agg_sanitized_indices["chapters"].add(offset + local_idx)
        for local_idx in result.get("paragraph_splits", []): agg_sanitized_indices["paragraphs"].add(offset + local_idx)
        for local_idx in result.get("semantic_splits", []): agg_sanitized_indices["semantics"].add(offset + local_idx)

    final_global_indices = {
        "chapter_splits": convert_sanitized_indices_to_global(sorted(list(agg_sanitized_indices["chapters"])), lookup_map),
        "paragraph_splits": convert_sanitized_indices_to_global(sorted(list(agg_sanitized_indices["paragraphs"])), lookup_map),
        "semantic_splits": convert_sanitized_indices_to_global(sorted(list(agg_sanitized_indices["semantics"])), lookup_map),
    }

    print("Step 4: Generating final text segments...")
    final_segments = {
        "chapter_segments": split_text_by_indices(original_text, final_global_indices["chapter_splits"]),
        "paragraph_segments": split_text_by_indices(original_text, final_global_indices["paragraph_splits"]),
        "semantic_segments": split_text_by_indices(original_text, final_global_indices["semantic_splits"]),
    }

    print("\n--- Final Analysis Results (as text segments) ---")
    for key, value in final_segments.items():
        print(f"\n--- {key} ({len(value)} segments) ---")
        if len(value) > 6:
            for i, segment in enumerate(value[:5]): print(f"[{i}]: {segment[:100].strip()}...")
            print("...")
            print(f"[{len(value)-1}]: {value[-1][:100].strip()}...")
        else:
            for i, segment in enumerate(value): print(f"[{i}]: {segment[:100].strip()}...")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "sample_text.txt"
    asyncio.run(main(filename=target_file))
