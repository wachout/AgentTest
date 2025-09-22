import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from dataclasses import asdict

from knowledge_extractor import create_knowledge_extractor

async def analyze_text(text_to_analyze: str, llm_provider: str = "deepseek"):
    """
    Analyzes a single piece of text and prints the extracted knowledge graph.
    """
    if not text_to_analyze:
        print("Error: No text provided for analysis.")
        print("Usage: python3 analyze_snippet.py \"<your text here>\" [--llm <deepseek|alibaba>]")
        return

    print("Loading environment...")
    load_dotenv()

    print(f"Initializing knowledge extractor with '{llm_provider}' provider...")
    extractor = create_knowledge_extractor(llm_provider=llm_provider)

    print("\n--- Extracting Knowledge from Snippet ---")
    knowledge_graph = await extractor.extract(text_to_analyze)

    print("\n--- Extraction Complete ---")
    if knowledge_graph:
        print(json.dumps(asdict(knowledge_graph), indent=2, ensure_ascii=False))
    else:
        print("Could not extract a knowledge graph from the provided text.")


if __name__ == "__main__":
    llm_provider = "deepseek"
    input_text = None

    args = sys.argv[1:]
    if args and not args[0].startswith('--'):
        input_text = args.pop(0)

    if "--llm" in args:
        try:
            llm_index = args.index("--llm")
            llm_provider = args[llm_index + 1]
            if llm_provider not in ["deepseek", "alibaba"]:
                print(f"Error: Invalid LLM provider '{llm_provider}'. Choose 'deepseek' or 'alibaba'.")
                sys.exit(1)
        except IndexError:
            print("Error: --llm flag requires an argument ('deepseek' or 'alibaba').")
            sys.exit(1)

    if input_text:
        asyncio.run(analyze_text(input_text, llm_provider=llm_provider))
    else:
        print("Error: No text provided.")
        print("Usage: python3 analyze_snippet.py \"<your text here>\" [--llm <deepseek|alibaba>]")
