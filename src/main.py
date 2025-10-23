import argparse
from src.agent import create_agent

def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The query to send to the agent")
    args = parser.parse_args()

    agent_executor = create_agent()

    response = agent_executor.invoke({"messages": [("user", args.query)]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()
