import json
import xml.etree.ElementTree as ET
import os
import glob

class VirtualNeo4j:
    """
    A class to simulate a Neo4j database in memory.
    It stores nodes and relationships in lists.
    """
    def __init__(self):
        self.nodes = []
        self.relationships = []
        print("VirtualNeo4j DB initialized.")

    def add_node(self, node_id, labels=None, properties=None):
        """
        Adds a node to the virtual database.
        """
        if labels is None:
            labels = []
        if properties is None:
            properties = {}

        node = {"id": node_id, "labels": labels, "properties": properties}
        self.nodes.append(node)
        # print(f"  - Added node: {node}")

    def add_relationship(self, source_id, target_id, rel_type, properties=None):
        """
        Adds a relationship to the virtual database.
        """
        if properties is None:
            properties = {}

        relationship = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "properties": properties
        }
        self.relationships.append(relationship)
        # print(f"  - Added relationship: {relationship}")

    def get_summary(self):
        """
        Returns a string summary of the database content.
        """
        return f"Database contains {len(self.nodes)} nodes and {len(self.relationships)} relationships."

    def __str__(self):
        return self.get_summary()

def load_from_graphml(file_path, db):
    """
    Loads nodes and relationships from a GraphML file into the virtual database.
    """
    print(f"-> Loading from GraphML file: {file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        key_map = {key.get('id'): key.get('attr.name') for key in root.findall('g:key', ns)}

        for node in root.findall('.//g:node', ns):
            node_id = node.get('id')
            labels = []
            properties = {}
            for data in node.findall('g:data', ns):
                key_id = data.get('key')
                attr_name = key_map.get(key_id)
                if attr_name == 'label':
                    labels.append(data.text)
                elif attr_name == 'properties':
                    properties = json.loads(data.text)
            db.add_node(node_id, labels=labels, properties=properties)

        for edge in root.findall('.//g:edge', ns):
            source_id = edge.get('source')
            target_id = edge.get('target')
            rel_type = "RELATED_TO"
            properties = {}
            for data in edge.findall('g:data', ns):
                key_id = data.get('key')
                attr_name = key_map.get(key_id)
                if attr_name == 'label':
                    rel_type = data.text
                elif attr_name == 'properties':
                    properties = json.loads(data.text)
            db.add_relationship(source_id, target_id, rel_type, properties=properties)
        print(f"   Finished loading from {file_path}.")
    except Exception as e:
        print(f"   Error loading GraphML file {file_path}: {e}")

def load_from_json(file_path, db):
    """
    Loads nodes and relationships from a JSON file into the virtual database.
    """
    print(f"-> Loading from JSON file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            # Check if file is empty
            content = f.read()
            if not content:
                print(f"   File {file_path} is empty. Skipping.")
                return
            data = json.loads(content)

        if "nodes" in data and data["nodes"]:
            for node in data["nodes"]:
                db.add_node(
                    node_id=node.get("id"),
                    labels=node.get("labels", []),
                    properties=node.get("properties", {})
                )

        if "relationships" in data and data["relationships"]:
            for rel in data["relationships"]:
                db.add_relationship(
                    source_id=rel.get("source"),
                    target_id=rel.get("target"),
                    rel_type=rel.get("type"),
                    properties=rel.get("properties", {})
                )
        print(f"   Finished loading from {file_path}.")
    except json.JSONDecodeError:
        print(f"   File {file_path} is not a valid JSON file. Skipping.")
    except Exception as e:
        print(f"   Error loading JSON file {file_path}: {e}")

def main_importer(data_directory="."):
    """
    Main function to orchestrate the import process.
    It scans the directory for specified files and loads them into a virtual DB.
    """
    print("Starting virtual import process...")
    db = VirtualNeo4j()

    # The list of files provided by the user
    files_to_import = [
        "graph_chunk_entity_relation.graphml",
        "output_graph.json",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_llm_response_cache.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json"
    ]

    for file_name in files_to_import:
        file_path = os.path.join(data_directory, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue

        if file_name.endswith(".graphml"):
            load_from_graphml(file_path, db)
        elif file_name.endswith(".json"):
            load_from_json(file_path, db)

    print("\nVirtual import process finished.")
    return db

if __name__ == '__main__':
    # This block demonstrates how to use the importer functions.
    # When you run this script directly, it will:
    # 1. Call the main_importer to load all data from the current directory.
    # 2. Print a summary of the imported data.
    # 3. Print the details of all nodes and relationships in the virtual database.

    print("--- Running Virtual Neo4j Importer Demo ---")
    # Run the main importer function for the current directory
    virtual_db = main_importer(data_directory=".")

    # Print the summary of the imported data
    print("\n--- Import Summary ---")
    print(virtual_db) # This will call the __str__ method

    # Optionally, print the details of nodes and relationships for verification
    print("\n--- Nodes Loaded ---")
    # Use json.dumps for pretty printing
    print(json.dumps(virtual_db.nodes, indent=2))

    print("\n--- Relationships Loaded ---")
    print(json.dumps(virtual_db.relationships, indent=2))

    print("\n--- Demo Finished ---")
