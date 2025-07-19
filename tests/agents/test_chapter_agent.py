import unittest
from unittest.mock import patch, MagicMock
from src.agents.chapter_agent import ChapterAgent, ChapterSplitResult, ChapterSplitPoint
import asyncio

class TestChapterAgent(unittest.TestCase):
    @patch('src.agents.chapter_agent.ChatOpenAI')
    def test_analyze(self, MockChatOpenAI):
        # Arrange
        mock_llm = MagicMock()
        mock_chain = MagicMock()

        async def mock_ainvoke(*args, **kwargs):
            return ChapterSplitResult(
                split_points=[
                    ChapterSplitPoint(position=0, title="第一章"),
                    ChapterSplitPoint(position=100, title="第二章"),
                ]
            )

        mock_chain.ainvoke.side_effect = mock_ainvoke

        mock_llm.with_structured_output.return_value = mock_chain
        MockChatOpenAI.return_value = mock_llm

        agent = ChapterAgent()
        agent.chain = mock_chain
        text = "This is a test text."

        # Act
        result = asyncio.run(agent.analyze(text))

        # Assert
        self.assertIsInstance(result, ChapterSplitResult)
        self.assertEqual(len(result.split_points), 2)
        self.assertEqual(result.split_points[0].title, "第一章")

if __name__ == "__main__":
    unittest.main()
