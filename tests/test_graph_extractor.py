import json
import unittest
from unittest.mock import patch, MagicMock

from lightrag.core.generator import GeneratorOutput
from lightrag.components.model_client.openai_client import OpenAIClient
from src.graph_extractor import (
    GraphExtractor,
    # We will still assert on the final KnowledgeGraph, Node, and Edge types
    KnowledgeGraph,
    Node,
    Edge,
)

# Mock client for initialization
class MockOpenAIClient(OpenAIClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_client = MagicMock()
        self.async_client = MagicMock()

class TestGraphExtractor(unittest.TestCase):
    """Unit tests for the GraphExtractor class, including chunking and merging."""

    def setUp(self):
        """Set up a reusable instance of the extractor for tests."""
        self.mock_model_client = MockOpenAIClient(api_key="test_key")

    @patch('lightrag.core.generator.Generator.call')
    def test_extract_with_chunking_and_merging(self, mock_generator_call):
        """
        Tests the full pipeline: raw string -> intermediate chunk graph -> final merged graph.
        """
        # 1. Arrange: Define input text and mock the raw string output from the generator

        long_text = "Part one has A and B. Part two has C and D."
        chunk1_text = "Part one has A and B."
        chunk2_text = "Part two has C and D."

        # Define the JSON data that matches the ChunkKnowledgeGraph schema
        graph_data_chunk1 = {
            "nodes": [{"id": 0, "label": "Entity", "properties": {"name": "A"}}, {"id": 1, "label": "Entity", "properties": {"name": "B"}}],
            "edges": [{"source_id": 0, "target_id": 1, "label": "RELATED_TO"}]
        }
        graph_data_chunk2 = {
            "nodes": [{"id": 0, "label": "Entity", "properties": {"name": "C"}}, {"id": 1, "label": "Entity", "properties": {"name": "D"}}],
            "edges": [{"source_id": 0, "target_id": 1, "label": "CONNECTED_TO"}]
        }

        # Simulate the raw string output from the LLM
        raw_output_chunk1 = f"Here is the graph: ```json\n{json.dumps(graph_data_chunk1)}\n```"
        raw_output_chunk2 = f"Here is the graph: ```json\n{json.dumps(graph_data_chunk2)}\n```"

        # The mock returns a GeneratorOutput object with the raw string
        output_chunk1 = GeneratorOutput(data=raw_output_chunk1)
        output_chunk2 = GeneratorOutput(data=raw_output_chunk2)

        mock_generator_call.side_effect = [output_chunk1, output_chunk2]

        # 2. Act: Call the method we want to test
        with patch('src.graph_extractor._chunk_text') as mock_chunk_text:
            mock_chunk_text.return_value = [chunk1_text, chunk2_text]

            extractor = GraphExtractor(model_client=self.mock_model_client)
            merged_graph = extractor.extract(long_text)

        # 3. Assert: Verify the final, merged graph is correct

        self.assertEqual(mock_generator_call.call_count, 2)

        # Check nodes
        self.assertEqual(len(merged_graph.nodes), 4)
        # Node A from chunk 1 (local id 0) -> global id 0
        self.assertEqual(merged_graph.nodes[0].id, 0)
        self.assertEqual(merged_graph.nodes[0].properties["name"], "A")
        self.assertEqual(merged_graph.nodes[0].chunk_id, 0)
        # Node C from chunk 2 (local id 0) -> global id 2
        self.assertEqual(merged_graph.nodes[2].id, 2)
        self.assertEqual(merged_graph.nodes[2].properties["name"], "C")
        self.assertEqual(merged_graph.nodes[2].chunk_id, 1)

        # Check edges
        self.assertEqual(len(merged_graph.edges), 2)
        # Edge from chunk 1 should have remapped global IDs (0 -> 1)
        self.assertEqual(merged_graph.edges[0].source_id, 0)
        self.assertEqual(merged_graph.edges[0].target_id, 1)
        self.assertEqual(merged_graph.edges[0].label, "RELATED_TO")
        # Edge from chunk 2 should have remapped global IDs (2 -> 3)
        self.assertEqual(merged_graph.edges[1].source_id, 2)
        self.assertEqual(merged_graph.edges[1].target_id, 3)
        self.assertEqual(merged_graph.edges[1].label, "CONNECTED_TO")

if __name__ == '__main__':
    unittest.main()