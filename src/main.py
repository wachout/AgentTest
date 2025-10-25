import argparse
from src.agent import create_agent

def main():
    """
    The main function for the command-line interface.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="AI agent that uses tools to answer questions.")
    parser.add_argument("query", type=str, help="The user's query for the agent.")

    # Parse the arguments
    args = parser.parse_args()

    # Create the agent
    agent = create_agent()

    # Get the agent's response
    inputs = {"query": args.query}
    response = agent.invoke(inputs)

    # Print the response
    print(response["result"])

if __name__ == "__main__":
    main()
