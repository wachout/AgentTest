import json
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError
from lightrag.core.component import Component
from lightrag.core.generator import Generator, GeneratorOutput
from lightrag.core.types import ModelType
from lightrag.core.functional import extract_json_str

# --- Intermediate Pydantic Models for LLM Output ---
# These models match the exact JSON structure we ask the LLM to produce.

class ChunkNode(BaseModel):
    """A node extracted from a single chunk, before global ID mapping."""
    id: int
    label: str
    properties: Optional[dict] = Field(default_factory=dict)

class ChunkEdge(BaseModel):
    """An edge extracted from a single chunk, with local node IDs."""
    source_id: int
    target_id: int
    label: str
    properties: Optional[dict] = Field(default_factory=dict)

class ChunkKnowledgeGraph(BaseModel):
    """The knowledge graph structure for a single chunk."""
    nodes: List[ChunkNode]
    edges: List[ChunkEdge]

# --- Final Pydantic Models for Merged Graph Output ---
# These models include the traceability fields (chunk_id, chunk_text).

class Node(BaseModel):
    """A node in the final, merged knowledge graph with global ID and chunk info."""
    id: int = Field(..., description="A globally unique identifier for the node across the entire document.")
    label: str
    properties: Optional[dict]
    chunk_id: int
    chunk_text: str

class Edge(BaseModel):
    """An edge in the final, merged knowledge graph with global IDs and chunk info."""
    source_id: int
    target_id: int
    label: str
    properties: Optional[dict]
    chunk_id: int
    chunk_text: str

class KnowledgeGraph(BaseModel):
    """The final, merged knowledge graph for the entire document."""
    nodes: List[Node]
    edges: List[Edge]

# --- LLM Prompt Template ---

EXTRACTION_PROMPT_TEMPLATE = """
Your task is to extract a knowledge graph from the provided text.
Identify all the meaningful entities as nodes and the relationships between them as edges.
The output must be a JSON object that strictly follows the provided schema.

Schema:
{
  "nodes": [
    {"id": <int>, "label": "<string>", "properties": {}},
    ...
  ],
  "edges": [
    {"source_id": <int>, "target_id": <int>, "label": "<string>", "properties": {}},
    ...
  ]
}

TEXT:
"{text}"
"""

# --- Helper function for text chunking ---
def _chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> List[str]:
    if len(text) <= chunk_size: return [text]
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
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs or {},
            template=EXTRACTION_PROMPT_TEMPLATE,
        )

    def _extract_from_chunk(self, chunk_text: str) -> Optional[ChunkKnowledgeGraph]:
        """Extracts a knowledge graph from a single text chunk into the intermediate format."""
        output: GeneratorOutput = self.generator.call(prompt_kwargs={"text": chunk_text})
        raw_response = output.data
        if not raw_response: return None
        try:
            json_str = extract_json_str(raw_response)
            # Parse into the intermediate ChunkKnowledgeGraph model
            return ChunkKnowledgeGraph.model_validate_json(json_str)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            print(f"Failed to parse knowledge graph from chunk. Error: {e}\nResponse: {raw_response}")
            return None

    def extract(self, text: str, chunk_size: int = 1024) -> KnowledgeGraph:
        """
        Extracts a knowledge graph from text, automatically handling chunking and merging.
        """
        chunks = _chunk_text(text, chunk_size=chunk_size)
        all_nodes: List[Node] = []
        all_edges: List[Edge] = []
        global_node_id_counter = 0
        node_map = {}

        for i, chunk in enumerate(chunks):
            chunk_graph = self._extract_from_chunk(chunk)
            if not chunk_graph or not chunk_graph.nodes: continue

            # Process nodes from the intermediate chunk_graph
            for chunk_node in chunk_graph.nodes:
                global_id = global_node_id_counter
                node_map[(i, chunk_node.id)] = global_id
                # Create the final Node object with enriched data
                all_nodes.append(
                    Node(
                        id=global_id,
                        label=chunk_node.label,
                        properties=chunk_node.properties,
                        chunk_id=i,
                        chunk_text=chunk,
                    )
                )
                global_node_id_counter += 1

            # Process edges
            if chunk_graph.edges:
                for chunk_edge in chunk_graph.edges:
                    global_source_id = node_map.get((i, chunk_edge.source_id))
                    global_target_id = node_map.get((i, chunk_edge.target_id))
                    if global_source_id is not None and global_target_id is not None:
                        # Create the final Edge object with remapped IDs and enriched data
                        all_edges.append(
                            Edge(
                                source_id=global_source_id,
                                target_id=global_target_id,
                                label=chunk_edge.label,
                                properties=chunk_edge.properties,
                                chunk_id=i,
                                chunk_text=chunk,
                            )
                        )
        return KnowledgeGraph(nodes=all_nodes, edges=all_edges)

    def call(self, input: str) -> Optional[ChunkKnowledgeGraph]:
        """
        Performs graph extraction on a single block of text without chunking.
        Note: This returns the raw chunk-level graph, not the final merged format.
        For most use cases, prefer the `extract` method.
        """
        return self._extract_from_chunk(input)