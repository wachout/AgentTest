import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from dataclasses import asdict
from typing import Dict, Any, Optional

from knowledge_extractor import create_knowledge_extractor

async def analyze_text(text_to_analyze: str, llm_provider: str = "deepseek") -> Optional[Dict[str, Any]]:
    """
    Core async function to analyze text and return a knowledge graph dictionary.
    """
    if not text_to_analyze:
        return None

    # This function is now designed to be called by other modules,
    # so we assume the environment is already loaded by the entry point.

    extractor = create_knowledge_extractor(llm_provider=llm_provider)
    knowledge_graph = await extractor.extract(text_to_analyze)

    if knowledge_graph:
        return asdict(knowledge_graph)
    return None

def analyze_snippet_sync(text_to_analyze: str, llm_provider: str = "deepseek") -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for the analysis function.
    This makes it easy to call from non-async code.
    It handles the asyncio event loop.
    """
    print("Initializing knowledge extractor...")
    # Environment variables are now expected to be set in the shell environment
    # before running the script.
    return asyncio.run(analyze_text(text_to_analyze, llm_provider))


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
        # Call the synchronous wrapper
        result = analyze_snippet_sync(input_text, llm_provider=llm_provider)

        print("\n--- Extraction Complete ---")
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Could not extract a knowledge graph from the provided text.")
    else:
        print("Error: No text provided.")
        print("Usage: python3 analyze_snippet.py \"<your text here>\" [--llm <deepseek|alibaba>]")
