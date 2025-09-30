import json
from knowledge_extractor import extract_knowledge
from graph_builder import build_graph_from_knowledge
import networkx as nx

def run_pipeline(text_filepath: str, output_knowledge_path: str, output_graph_path: str):
    """
    Runs the full knowledge extraction and graph building pipeline.

    Args:
        text_filepath: Path to the input text file.
        output_knowledge_path: Path to save the extracted knowledge JSON.
        output_graph_path: Path to save the final graph file.
    """
    print(f"--- Starting Pipeline ---")

    # Step 1: Read the input text
    try:
        with open(text_filepath, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Successfully read text from '{text_filepath}'")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{text_filepath}'")
        return

    # Step 2: Extract knowledge using the LLM
    print("Extracting knowledge from text...")
    knowledge = extract_knowledge(text)
    if not knowledge:
        print("Knowledge extraction failed. Aborting.")
        return

    with open(output_knowledge_path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved extracted knowledge to '{output_knowledge_path}'")

    # Step 3: Build the graph
    print("Building knowledge graph...")
    graph = build_graph_from_knowledge(knowledge)

    # Step 4: Save the graph
    nx.write_graphml(graph, output_graph_path)
    print(f"Successfully built and saved knowledge graph to '{output_graph_path}'")

    # Step 5: Display summary
    print("\n--- Pipeline Complete ---")
    print("Summary:")
    print(f"  - Extracted {len(knowledge.get('entities', []))} entities.")
    print(f"  - Extracted {len(knowledge.get('keywords', []))} keywords.")
    print(f"  - Extracted {len(knowledge.get('relationships', []))} relationships.")
    print(f"  - Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print("-------------------------")


if __name__ == "__main__":
    INPUT_TEXT_FILE = "wandering_earth_summary.txt"
    OUTPUT_KNOWLEDGE_FILE = "knowledge.json"
    OUTPUT_GRAPH_FILE = "knowledge_graph.graphml"

    run_pipeline(
        text_filepath=INPUT_TEXT_FILE,
        output_knowledge_path=OUTPUT_KNOWLEDGE_FILE,
        output_graph_path=OUTPUT_GRAPH_FILE
    )