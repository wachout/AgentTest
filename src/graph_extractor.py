from typing import List, Optional

from pydantic import BaseModel, Field
from lightrag.core.component import Component
from lightrag.core.generator import Generator
from lightrag.core.types import ModelType

# --- Pydantic Models for Graph Structure ---

class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: int = Field(..., description="A unique identifier for the node. In the final merged graph, this ID will be globally unique.")
    label: str = Field(..., description="The primary label or name of the node (e.g., 'Person', 'Organization').")
    properties: Optional[dict] = Field(default_factory=dict, description="A dictionary of properties for the node.")
    chunk_id: int = Field(..., description="The ID of the text chunk from which this node was extracted.")
    chunk_text: str = Field(..., description="The raw text of the chunk from which this node was extracted.")

class Edge(BaseModel):
    """Represents a directed edge between two nodes in the knowledge graph."""
    source_id: int = Field(..., description="The globally unique identifier of the source node.")
    target_id: int = Field(..., description="The globally unique identifier of the target node.")
    label: str = Field(..., description="The type or name of the relationship (e.g., 'WORKS_FOR', 'LOCATED_IN').")
    properties: Optional[dict] = Field(default_factory=dict, description="A dictionary of properties for the edge.")
    chunk_id: int = Field(..., description="The ID of the text chunk from which this edge was extracted.")
    chunk_text: str = Field(..., description="The raw text of the chunk from which this edge was extracted.")

class KnowledgeGraph(BaseModel):
    """Represents the entire knowledge graph with a list of nodes and edges."""
    nodes: List[Node] = Field(..., description="A list of all nodes in the graph.")
    edges: List[Edge] = Field(..., description="A list of all edges connecting the nodes.")

# --- LLM Prompt Template ---

EXTRACTION_PROMPT_TEMPLATE = """
Your task is to extract a knowledge graph from the provided text.
Identify all the meaningful entities as nodes and the relationships between them as edges.

- Each node must have a unique integer ID local to this text, a descriptive label, and properties.
- Each edge must connect two nodes using their local IDs and have a label describing the relationship.
- The output must be a JSON object that strictly follows the provided schema for a KnowledgeGraph containing nodes and edges.
- Do not add any extra fields like chunk_id or chunk_text; they will be added later.

TEXT:
"{text}"
"""

# --- Helper function for text chunking ---
def _chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> List[str]:
    """Splits text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

class GraphExtractor(Component):
    """
    Extracts a knowledge graph from text, handling long documents by splitting them into chunks.
    """
    def __init__(self, model_client: ModelType, model_kwargs: Optional[dict] = None):
        super().__init__()
        self.model_kwargs = model_kwargs or {}
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=self.model_kwargs
        )

    def _extract_from_chunk(self, chunk_text: str) -> KnowledgeGraph:
        """Extracts a knowledge graph from a single text chunk."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=chunk_text)
        # The Pydantic model here is used for parsing the LLM output for a single chunk
        # It does not yet contain the chunk_id or chunk_text fields.
        response = self.generator.call(prompt=prompt, output_cls=KnowledgeGraph)
        return response

    def extract(self, text: str, chunk_size: int = 1024) -> KnowledgeGraph:
        """
        Extracts a knowledge graph from text, automatically handling chunking and merging.

        Args:
            text: The input text to process.
            chunk_size: The size of text chunks to split the document into.

        Returns:
            A single, merged KnowledgeGraph containing all nodes and edges from the document.
        """
        chunks = _chunk_text(text, chunk_size=chunk_size)

        all_nodes: List[Node] = []
        all_edges: List[Edge] = []

        global_node_id_counter = 0
        node_map = {}  # Maps (chunk_id, local_node_id) to global_node_id

        for i, chunk in enumerate(chunks):
            chunk_graph = self._extract_from_chunk(chunk)

            if not chunk_graph or not chunk_graph.nodes:
                continue

            # Process nodes from the current chunk
            for node in chunk_graph.nodes:
                # Assign a new global ID
                global_id = global_node_id_counter
                node_map[(i, node.id)] = global_id

                # Create the new node with global ID and chunk info
                all_nodes.append(
                    Node(
                        id=global_id,
                        label=node.label,
                        properties=node.properties,
                        chunk_id=i,
                        chunk_text=chunk,
                    )
                )
                global_node_id_counter += 1

            # Process edges from the current chunk
            if chunk_graph.edges:
                for edge in chunk_graph.edges:
                    # Remap source and target IDs to global IDs
                    global_source_id = node_map.get((i, edge.source_id))
                    global_target_id = node_map.get((i, edge.target_id))

                    if global_source_id is not None and global_target_id is not None:
                        all_edges.append(
                            Edge(
                                source_id=global_source_id,
                                target_id=global_target_id,
                                label=edge.label,
                                properties=edge.properties,
                                chunk_id=i,
                                chunk_text=chunk,
                            )
                        )

        return KnowledgeGraph(nodes=all_nodes, edges=all_edges)

    # Maintain the 'call' method for backward compatibility and simple, single-chunk extractions.
    def call(self, input: str) -> KnowledgeGraph:
        """
        Performs graph extraction on a single block of text without chunking.
        For automatic chunking, use the `extract` method.
        """
        return self._extract_from_chunk(input)