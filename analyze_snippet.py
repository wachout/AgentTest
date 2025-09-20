import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from dataclasses import asdict

from knowledge_extractor import create_knowledge_extractor

async def analyze_text(text_to_analyze: str):
    """
    Analyzes a single piece of text and prints the extracted knowledge graph.
    """
    if not text_to_analyze:
        print("Error: No text provided for analysis.")
        print("Usage: python3 analyze_snippet.py \"<your text here>\"")
        return

    print("Loading environment and setting up API clients...")
    load_dotenv()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("ERROR: DEEPSEEK_API_KEY not set in .env file.")
        return

    # Set environment variables for the OpenAI client library used by lightrag
    os.environ["OPENAI_API_KEY"] = deepseek_api_key
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

    print("Initializing knowledge extractor...")
    extractor = create_knowledge_extractor()

    print("\n--- Extracting Knowledge from Snippet ---")
    knowledge_graph = await extractor.extract(text_to_analyze)

    print("\n--- Extraction Complete ---")
    if knowledge_graph:
        # Convert dataclass to dict and print as formatted JSON
        print(json.dumps(asdict(knowledge_graph), indent=2, ensure_ascii=False))
    else:
        print("Could not extract a knowledge graph from the provided text.")


if __name__ == "__main__":
    # Get text from the first command-line argument
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        asyncio.run(analyze_text(input_text))
    else:
        print("Error: No text provided.")
        print("Usage: python3 analyze_snippet.py \"<your text here>\"")
