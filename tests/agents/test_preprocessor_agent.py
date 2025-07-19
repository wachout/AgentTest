import unittest
from src.agents.preprocessor_agent import PreprocessorAgent

class TestPreprocessorAgent(unittest.TestCase):
    def test_short_text(self):
        agent = PreprocessorAgent(max_chunk_size=20000)
        text = "This is a short text."
        chunks = agent.process(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text(self):
        agent = PreprocessorAgent(max_chunk_size=100, overlap=10)
        text = "a" * 250
        chunks = agent.process(text)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].endswith("a" * 10))
        self.assertTrue(chunks[1].startswith("a" * 10))
        self.assertTrue(chunks[1].endswith("a" * 10))
        self.assertTrue(chunks[2].startswith("a" * 10))

if __name__ == "__main__":
    unittest.main()
