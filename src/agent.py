import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent as create_langgraph_agent
from src.tools import search_tool

load_dotenv()

def create_agent():
    """Creates and returns a ReAct agent."""

    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    # Initialize the language model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    # Define the tools the agent can use
    tools = [search_tool]

    # Create the agent executor
    agent_executor = create_langgraph_agent(llm, tools)

    return agent_executor
