import os
from dotenv import load_dotenv

from lightrag.components.model_client.openai_client import OpenAIClient
from src.graph_extractor import GraphExtractor

# A long text example that is likely to be split into multiple chunks.
# It contains distinct facts in different sections to test the merging logic.
LONG_TEXT_EXAMPLE = """
In the early days of personal computing, Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. Their first product, the Apple I, was a simple circuit board. This was followed by the Apple II, which became a massive success and helped popularize the personal computer. Steve Jobs, known for his visionary leadership, was ousted from the company in 1985 but returned in 1997 to save it from near bankruptcy.

Meanwhile, on the other side of the country, Microsoft was founded by Bill Gates and Paul Allen in 1975. They started by developing a BASIC interpreter for the Altair 8800. Their big break came when they licensed their operating system, MS-DOS, to IBM for the first IBM PC. This partnership established Microsoft as a dominant force in the software industry. Bill Gates served as the CEO until 2000, succeeded by Steve Ballmer.

Later, in the age of the internet, a new giant emerged. Google was founded by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University. They officially incorporated the company in 1998. Their innovative search engine quickly outcompeted rivals. Larry Page served as the first CEO, with Eric Schmidt later taking on the role to provide more mature leadership for the rapidly growing company. Google is headquartered in Mountain View, California.
"""

def main():
    """
    Demonstrates how to use the GraphExtractor to extract a knowledge graph from text,
    including handling long documents with the new chunking functionality.
    """
    load_dotenv()

    print("--- Running Graph Extraction with DeepSeek (Chunking Enabled) ---")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("DeepSeek API key not found. Skipping DeepSeek example.")
        print("Please set the DEEPSEEK_API_KEY environment variable in a .env file.\n")
    else:
        deepseek_client = OpenAIClient(api_key=deepseek_api_key)
        deepseek_client.sync_client.base_url = "https://api.deepseek.com/v1"

        extractor = GraphExtractor(
            model_client=deepseek_client,
            model_kwargs={"model": "deepseek-reasoner"}
        )

        print("Input text is a long document about the history of Apple, Microsoft, and Google.")
        print(f"Text length: {len(LONG_TEXT_EXAMPLE)} characters.")

        try:
            # Use the `extract` method to automatically handle chunking.
            # We can specify a chunk_size, but we'll use the default for this example.
            knowledge_graph = extractor.extract(LONG_TEXT_EXAMPLE)

            print("\nExtracted Knowledge Graph (DeepSeek):")
            print("Note the 'chunk_id' and 'chunk_text' fields in the nodes and edges.")
            print(knowledge_graph.model_dump_json(indent=2))
        except Exception as e:
            print(f"An error occurred during DeepSeek extraction: {e}")

if __name__ == "__main__":
    main()