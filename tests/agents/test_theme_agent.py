import unittest
from unittest.mock import patch, MagicMock
from src.agents.theme_agent import ThemeAgent, ThemeSplitResult, ThemeSplitPoint
import asyncio

class TestThemeAgent(unittest.TestCase):
    @patch('src.agents.theme_agent.ChatOpenAI')
    def test_analyze(self, MockChatOpenAI):
        # Arrange
        mock_llm = MagicMock()
        mock_chain = MagicMock()

        async def mock_ainvoke(*args, **kwargs):
            return ThemeSplitResult(
                split_points=[
                    ThemeSplitPoint(position=0, theme="总则"),
                    ThemeSplitPoint(position=100, theme="财务机构"),
                ]
            )

        mock_chain.ainvoke.side_effect = mock_ainvoke

        mock_llm.with_structured_output.return_value = mock_chain
        MockChatOpenAI.return_value = mock_llm

        agent = ThemeAgent()
        agent.chain = mock_chain
        text = "This is a test text."

        # Act
        result = asyncio.run(agent.analyze(text))

        # Assert
        self.assertIsInstance(result, ThemeSplitResult)
        self.assertEqual(len(result.split_points), 2)
        self.assertEqual(result.split_points[0].theme, "总则")

if __name__ == "__main__":
    unittest.main()
