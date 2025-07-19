import unittest
from src.agents.aggregator_agent import AggregatorAgent
from src.agents.metadata_agent import ArticleMetadata
from src.agents.chapter_agent import ChapterSplitResult, ChapterSplitPoint
from src.agents.theme_agent import ThemeSplitResult, ThemeSplitPoint

class TestAggregatorAgent(unittest.TestCase):
    def test_aggregate(self):
        agent = AggregatorAgent()
        results = [
            {
                "text": "a" * 2000,
                "metadata": ArticleMetadata(
                    article_type="新闻",
                    business_type="科技",
                    creation_date="2024-01-01",
                    update_date="2024-01-02",
                    expiration_date="N/A",
                    region="省级"
                ),
                "chapters": ChapterSplitResult(
                    split_points=[ChapterSplitPoint(position=10, title="第一章")]
                ),
                "themes": ThemeSplitResult(
                    split_points=[ThemeSplitPoint(position=20, theme="引言")]
                ),
            },
            {
                "text": "b" * 2000,
                "metadata": ArticleMetadata(
                    article_type="新闻",
                    business_type="科技",
                    creation_date="2024-01-01",
                    update_date="2024-01-02",
                    expiration_date="N/A",
                    region="省级"
                ),
                "chapters": ChapterSplitResult(
                    split_points=[ChapterSplitPoint(position=30, title="第二章")]
                ),
                "themes": ThemeSplitResult(
                    split_points=[ThemeSplitPoint(position=40, theme="核心")]
                ),
            },
        ]

        aggregated_result = agent.aggregate(results)

        self.assertEqual(aggregated_result["metadata"].article_type, "新闻")
        self.assertEqual(len(aggregated_result["chapters"]), 2)
        self.assertEqual(aggregated_result["chapters"][0]["position"], 10)
        self.assertEqual(aggregated_result["chapters"][1]["position"], 30 + 2000 - 1000)
        self.assertEqual(len(aggregated_result["themes"]), 2)
        self.assertEqual(aggregated_result["themes"][0]["position"], 20)
        self.assertEqual(aggregated_result["themes"][1]["position"], 40 + 2000 - 1000)

if __name__ == "__main__":
    unittest.main()
