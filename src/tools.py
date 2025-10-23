import os
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool

load_dotenv()

@tool
def search_tool(query: str) -> str:
    """A tool that can be used to search the web for a given query."""
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        raise ValueError("SERPAPI_API_KEY not found in .env file")

    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    return search.run(query)
