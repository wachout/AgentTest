import asyncio
import os
from pprint import pprint
from dotenv import load_dotenv

# Import the graph builder function from our other module
from graph_builder import create_graph, AgentState

async def main():
    """
    The main execution function for our multi-agent text splitting system.
    """
    # Load environment variables from .env file
    # This will load the DEEPSEEK_API_KEY
    load_dotenv()

    # Check if the API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        print("Please create a .env file and add your key.")
        return

    print("Reading sample text...")
    try:
        with open("sample_text.txt", "r", encoding="utf-8") as f:
            text_content = f.read()
    except FileNotFoundError:
        print("ERROR: sample_text.txt not found. Please create it.")
        return

    # Create the compiled LangGraph application
    graph_app = create_graph()

    # Define the initial state to be passed to the graph
    # The graph will update the other fields as it runs.
    initial_state: AgentState = {
        "input_text": text_content,
        "chapter_splits": [],
        "paragraph_splits": [],
        "semantic_splits": [],
        "error": None,
    }

    print("\n--- Invoking Multi-Agent Graph ---")
    # Asynchronously invoke the graph with our initial state
    final_state = await graph_app.ainvoke(initial_state)

    print("\n--- Graph Execution Complete ---")

    if final_state.get("error"):
        print("\nAn error occurred during graph execution:")
        print(final_state["error"])
    else:
        print("\nFinal analysis results:")
        # Use pprint for a more readable dictionary output
        pprint(
            {
                "chapter_splits": final_state.get("chapter_splits"),
                "paragraph_splits": final_state.get("paragraph_splits"),
                "semantic_splits": final_state.get("semantic_splits"),
            }
        )

# Standard Python entry point for running async functions
if __name__ == "__main__":
    asyncio.run(main())
