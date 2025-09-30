import json
import networkx as nx

def build_graph_from_knowledge(knowledge: dict) -> nx.DiGraph:
    """
    Builds a NetworkX directed graph from the extracted knowledge.

    Args:
        knowledge: A dictionary containing entities and relationships.

    Returns:
        A NetworkX DiGraph object.
    """
    G = nx.DiGraph()

    # Add nodes (entities) with their attributes
    if "entities" in knowledge:
        for entity in knowledge["entities"]:
            node_name = entity.get("name")
            if node_name:
                # The 'attributes' key contains a dict of attributes
                attributes = entity.get("attributes", {})
                # Add the 'type' to the attributes as well
                attributes['type'] = entity.get("type", "Unknown")
                G.add_node(node_name, **attributes)

    # Add edges (relationships)
    if "relationships" in knowledge:
        for rel in knowledge["relationships"]:
            source = rel.get("source")
            target = rel.get("target")
            relation = rel.get("relation", "")

            # Ensure both source and target nodes exist before adding an edge
            if source in G and target in G:
                G.add_edge(source, target, label=relation)

    return G

if __name__ == "__main__":
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            extracted_knowledge = json.load(f)

        knowledge_graph = build_graph_from_knowledge(extracted_knowledge)

        # Save the graph to a file in GraphML format
        # This format is portable and can be used with tools like Gephi
        graph_file = "knowledge_graph.graphml"
        nx.write_graphml(knowledge_graph, graph_file)

        print(f"Successfully built the knowledge graph and saved it to '{graph_file}'")
        print("\n--- Graph Info ---")
        print(f"Number of nodes: {knowledge_graph.number_of_nodes()}")
        print(f"Number of edges: {knowledge_graph.number_of_edges()}")
        print("--------------------")

    except FileNotFoundError:
        print("Error: 'knowledge.json' not found. Please run the knowledge extraction step first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")