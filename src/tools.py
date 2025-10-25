from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str):
    """
    Searches the knowledge base for an answer to the user's query.
    """
    return f"Searching the knowledge base for: '{query}'"

@tool
def query_database(query: str):
    """
    Queries the database to retrieve statistical information.
    """
    return f"Querying the database for: '{query}'"

@tool
def perform_data_analysis(query: str):
    """
    Performs in-depth data analysis and generates a report.
    """
    return f"Performing data analysis for: '{query}'"
