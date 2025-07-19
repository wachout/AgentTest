import unittest
from unittest.mock import patch, MagicMock
from src.agents.metadata_agent import MetadataAgent, ArticleMetadata
import asyncio

class TestMetadataAgent(unittest.TestCase):
    @patch('src.agents.metadata_agent.ChatOpenAI')
    def test_analyze(self, MockChatOpenAI):
        # Arrange
        mock_llm = MagicMock()
        mock_chain = MagicMock()

        # Configure the mock chain's ainvoke to be an async function
        async def mock_ainvoke(*args, **kwargs):
            return ArticleMetadata(
                article_type="新闻",
                business_type="科技",
                creation_date="2024-01-01",
                update_date="2024-01-02",
                expiration_date="N/A",
                region="省级"
            )

        # Arrange
        mock_llm = MagicMock()
        mock_chain = MagicMock()

        # Configure the mock chain's ainvoke to be an async function
        async def mock_ainvoke(*args, **kwargs):
            return ArticleMetadata(
                article_type="新闻",
                business_type="科技",
                creation_date="2024-01-01",
                update_date="2024-01-02",
                expiration_date="N/A",
                region="省级"
            )

        mock_chain.ainvoke.side_effect = mock_ainvoke

        mock_llm.with_structured_output.return_value = mock_chain
        MockChatOpenAI.return_value = mock_llm

        agent = MetadataAgent()
        agent.chain = mock_chain  # Patch the instance's chain
        text = "This is a test text."

        # Act
        result = asyncio.run(agent.analyze(text))

        # Assert
        self.assertIsInstance(result, ArticleMetadata)
        self.assertEqual(result.article_type, "新闻")

if __name__ == "__main__":
    unittest.main()
