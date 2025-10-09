import unittest
from unittest.mock import patch, MagicMock

from lightrag.components.model_client.openai_client import OpenAIClient
from src.graph_extractor import (
    GraphExtractor,
    KnowledgeGraph,
    Node,
    Edge,
)

# Mock client for initialization
class MockOpenAIClient(OpenAIClient):
    def __init__(self, **kwargs):
        pass

class TestGraphExtractor(unittest.TestCase):
    """Unit tests for the GraphExtractor class, including chunking and merging."""

    def setUp(self):
        """Set up a reusable instance of the extractor for tests."""
        self.mock_model_client = MockOpenAIClient(api_key="test_key")
        self.extractor = GraphExtractor(model_client=self.mock_model_client)

    @patch('src.graph_extractor.GraphExtractor._extract_from_chunk')
    def test_extract_with_chunking_and_merging(self, mock_extract_from_chunk):
        """
        Tests that the extract method correctly chunks text, extracts from each chunk,
        and merges the results into a single, coherent knowledge graph.
        """
        # 1. Arrange: Define the input text and mock the chunk-level extractions

        long_text = "Part one of the story. Part two of the story."
        # Let's assume our chunking logic splits this into two chunks
        chunk1_text = "Part one of the story."
        chunk2_text = "Part two of the story."

        # Mock the return value for the first chunk
        graph_chunk1 = KnowledgeGraph(
            nodes=[
                Node(id=0, label="Entity", properties={"name": "A"}, chunk_id=0, chunk_text=""), # chunk info will be ignored
                Node(id=1, label="Entity", properties={"name": "B"}, chunk_id=0, chunk_text="")
            ],
            edges=[Edge(source_id=0, target_id=1, label="RELATED_TO", chunk_id=0, chunk_text="")]
        )

        # Mock the return value for the second chunk
        graph_chunk2 = KnowledgeGraph(
            nodes=[
                Node(id=0, label="Entity", properties={"name": "C"}, chunk_id=0, chunk_text=""),
                Node(id=1, label="Entity", properties={"name": "D"}, chunk_id=0, chunk_text="")
            ],
            edges=[Edge(source_id=0, target_id=1, label="CONNECTED_TO", chunk_id=0, chunk_text="")]
        )

        # The mock will return graph_chunk1 on the first call, and graph_chunk2 on the second.
        mock_extract_from_chunk.side_effect = [graph_chunk1, graph_chunk2]

        # 2. Act: Call the method we want to test
        # We patch the chunking function to have a predictable input to the extractor
        with patch('src.graph_extractor._chunk_text') as mock_chunk_text:
            mock_chunk_text.return_value = [chunk1_text, chunk2_text]
            merged_graph = self.extractor.extract(long_text)

        # 3. Assert: Verify the merged graph is correct

        # Check that the chunk extraction was called for each chunk
        self.assertEqual(mock_extract_from_chunk.call_count, 2)
        mock_extract_from_chunk.assert_any_call(chunk1_text)
        mock_extract_from_chunk.assert_any_call(chunk2_text)

        # Check the merged nodes
        self.assertEqual(len(merged_graph.nodes), 4)
        # Node A from chunk 1 (local id 0) should now have global id 0
        self.assertEqual(merged_graph.nodes[0].id, 0)
        self.assertEqual(merged_graph.nodes[0].properties["name"], "A")
        self.assertEqual(merged_graph.nodes[0].chunk_id, 0)
        self.assertEqual(merged_graph.nodes[0].chunk_text, chunk1_text)
        # Node C from chunk 2 (local id 0) should now have global id 2
        self.assertEqual(merged_graph.nodes[2].id, 2)
        self.assertEqual(merged_graph.nodes[2].properties["name"], "C")
        self.assertEqual(merged_graph.nodes[2].chunk_id, 1)
        self.assertEqual(merged_graph.nodes[2].chunk_text, chunk2_text)

        # Check the merged edges
        self.assertEqual(len(merged_graph.edges), 2)
        # Edge from chunk 1 should have remapped global IDs (0 -> 1)
        self.assertEqual(merged_graph.edges[0].source_id, 0)
        self.assertEqual(merged_graph.edges[0].target_id, 1)
        self.assertEqual(merged_graph.edges[0].label, "RELATED_TO")
        self.assertEqual(merged_graph.edges[0].chunk_id, 0)
        # Edge from chunk 2 should have remapped global IDs (2 -> 3)
        self.assertEqual(merged_graph.edges[1].source_id, 2)
        self.assertEqual(merged_graph.edges[1].target_id, 3)
        self.assertEqual(merged_graph.edges[1].label, "CONNECTED_TO")
        self.assertEqual(merged_graph.edges[1].chunk_id, 1)

if __name__ == '__main__':
    unittest.main()