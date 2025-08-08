from graph import Graph

def main():
    g = Graph()

    g.add_node("A")
    g.add_node("B")
    g.add_node("C")
    g.add_node("D")
    g.add_node("E")
    g.add_node("F")

    g.add_edge("A", "B", 1, "A to B")
    g.add_edge("A", "C", 4, "A to C")
    g.add_edge("B", "D", 2, "B to D")
    g.add_edge("C", "E", 5, "C to E")
    g.add_edge("D", "E", 1, "D to E")
    g.add_edge("D", "F", 3, "D to F")
    g.add_edge("E", "F", 1, "E to F")

    print("Shortest path from A to F (Dijkstra):")
    for step in g.dijkstra("A", "F"):
        print(f"  {step['from']} -> {step['to']} (weight: {step['weight']}, desc: {step['description']})")

    nodes_to_visit = ["A", "D", "F"]
    print(f"\nOptimal path for nodes {nodes_to_visit}:")
    optimal_path = g.find_optimal_path_for_nodes(nodes_to_visit)
    if optimal_path:
        for step in optimal_path:
            print(f"  {step['from']} -> {step['to']} (weight: {step['weight']}, desc: {step['description']})")

if __name__ == "__main__":
    main()
