import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

# Load environment variables from .env file
load_dotenv()

# Set the OPENAI_API_KEY environment variable for LangChain compatibility
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "dummy_key")

# 1. Define Graph Schema
graph_schema = """
Node Properties:
- entity_id: string (The name of the node, e.g., '潮汐')
- source_id: string (The ID of the source paragraph)
- description: string (A description of the node's context)
- entity_type: string (The type of the entity)
- file_path: string (The path to the source file)

Edge Properties (on the 'relation' variable):
- weight: float
- description: string
- keywords: string (Comma-separated keywords)
- source_id: string
"""

# 2. Configure the LLM
llm = ChatOpenAI(
    temperature=0.0,
    model="qwen3-32b",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False}
)

# 3. Build the LangGraph Agent

# 3.1 Define the State
class AgentState(TypedDict):
    query: str
    nodes: List[str]
    relationships: Optional[Dict[str, Any]]
    start_node: str
    end_node: Optional[str]
    cypher_query: str
    query_result: Optional[List[dict]]
    query_error: Optional[str]
    errors: List[str]
    refine_count: int

# 3.2 Define the Nodes

# Node 1: Smart Analysis
class AnalysisResult(BaseModel):
    start_node: str = Field(description="The primary start node, selected from the candidate list.")
    end_node: Optional[str] = Field(description="The end node if a relationship is implied, selected from the candidate list.")

def analyze_query_and_nodes(state: AgentState):
    """Analyzes the query to select start and end nodes from the candidates."""
    print("---ANALYZING QUERY AND NODES---")

    prompt_analyzer = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an intelligent analysis engine. Your task is to understand a user's query and identify the start and end nodes from a given list of candidate nodes. "
             "The query's structure will indicate the direction. For example, in 'What is related to Node A?', 'Node A' is the start_node. "
             "In 'How does Node A relate to Node B?', 'Node A' is the start_node and 'Node B' is the end_node. "
             "Respond with a JSON object: {{'start_node': '...', 'end_node': '...'}}."
            ),
            ("human", "Query: {query}\nCandidate Nodes: {nodes}"),
        ]
    )
    analyzer_runnable = prompt_analyzer | llm.with_structured_output(AnalysisResult)

    analysis = analyzer_runnable.invoke({
        "query": state["query"],
        "nodes": state["nodes"]
    })

    print(f"Analysis complete: Start Node='{analysis.start_node}', End Node='{analysis.end_node}'")
    return {
        "start_node": analysis.start_node,
        "end_node": analysis.end_node,
        "refine_count": 0,
        "errors": [],
    }


# Node 2: Generate Cypher Query (Restored LLM Generation with a Robust Prompt)
prompt_generate_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query generator. Your task is to write a Cypher query based on the provided start/end nodes and optional relationship filters. "
         "The query must be read-only.\n"
         "Graph Schema:\n{schema}\n"
         "---"
         "**CRITICAL RULES**:\n"
         "1.  **Variable Names**: The starting node MUST be named `start_node`. If there is an ending node, it MUST be named `end_node`. The relationship MUST be named `relation`.\n"
         "2.  **Node Matching**: Always match nodes by their `entity_id` property (e.g., `start_node.entity_id = '{start_node_name}'`).\n"
         "3.  **Return Values**: ALWAYS return the full node and relationship objects (e.g., `RETURN start_node, relation, end_node`).\n"
         "4.  **Relationship Filters**: If `relationship_filters` are provided, you MUST add a `WHERE` clause to filter the `relation` variable. For string properties like `keywords`, use the `CONTAINS` operator for partial matches. For other types, use `=`.\n"
         "5.  **No End Node**: If `end_node` is not provided, find any connected node and alias it as `end_node`.\n"
         "---"
         "**Example 1: Find relationship between two nodes with relationship filters.**\n"
         "Inputs:\n"
         "  - start_node: '流浪地球'\n"
         "  - end_node: '刹车时代'\n"
         "  - relationship_filters: {{'keywords': '事件描述'}}\n"
         "Generated Cypher: MATCH (start_node {{entity_id: '流浪地球'}})-[relation]-(end_node {{entity_id: '刹车时代'}}) WHERE relation.keywords CONTAINS '事件描述' RETURN start_node, relation, end_node\n"
         "---"
         "**Example 2: Find all connected nodes to a single start node.**\n"
         "Inputs:\n"
         "  - start_node: '潮汐'\n"
         "  - end_node: null\n"
         "  - relationship_filters: null\n"
         "Generated Cypher: MATCH (start_node {{entity_id: '潮汐'}})-[relation]-(end_node) RETURN start_node, relation, end_node"
        ),
        ("human",
         "Generate a Cypher query with these parameters:\n"
         "Start Node: {start_node}\n"
         "End Node: {end_node}\n"
         "Relationship Filters: {relationship_filters}"
        ),
    ]
)
query_generator_runnable = prompt_generate_query | llm

def generate_query(state: AgentState):
    """Generates a Cypher query using the LLM with a robust prompt."""
    print("---GENERATING CYPHER QUERY (LLM)---")

    generation = query_generator_runnable.invoke({
        "schema": graph_schema,
        "start_node": state["start_node"],
        "end_node": state["end_node"],
        "relationship_filters": state.get("relationships")
    })

    cypher_query = generation.content.strip()
    if "```cypher" in cypher_query:
        cypher_query = cypher_query.split("```cypher")[1].split("```")[0].strip()
    elif "```" in cypher_query:
        cypher_query = cypher_query.split("```")[1].strip()

    print(f"Generated Query: {cypher_query}")
    return {"cypher_query": cypher_query}

# Node 3: Check Cypher Query (Restored)
class QueryCheck(BaseModel):
    is_correct: bool
    errors: List[str]

prompt_check_query = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Neo4j expert. Check if the following Cypher query is syntactically correct. Respond with a JSON object: {{'is_correct': true, 'errors': []}} or {{'is_correct': false, 'errors': ['error message']}}."),
        ("human", "Query: {cypher_query}"),
    ]
)
query_checker_runnable = prompt_check_query | llm.with_structured_output(QueryCheck)

def check_query(state: AgentState):
    print("---CHECKING CYPHER QUERY---")
    check_result = query_checker_runnable.invoke({"cypher_query": state["cypher_query"]})
    if check_result.is_correct:
        print("Query Check: CORRECT")
        return {"errors": []}
    else:
        print(f"Query Check: INCORRECT. Errors: {check_result.errors}")
        return {"errors": check_result.errors}

# Node 4: Execute Query
def execute_query(state: AgentState):
    print("---EXECUTING CYPHER QUERY---")
    # ... (rest of the function is unchanged)
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]) or "YOUR_NEO4J_URI" in uri:
        error_msg = "Neo4j connection details are not configured in .env file."
        print(error_msg)
        return {"query_error": error_msg, "query_result": []}

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(state["cypher_query"])
            records = [record.data() for record in result]
            print(f"Query Result: {records}")
            return {"query_result": records, "query_error": None}
    except Exception as e:
        error_msg = f"Failed to execute query: {e}"
        print(error_msg)
        return {"query_error": error_msg, "query_result": []}
    finally:
        driver.close()


# 3.4 Assemble the graph
def decide_to_execute(state: AgentState):
    if not state.get("errors"):
        return "execute"
    else:
        # Simplified: no refinement loop for this final version
        return END

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_query_and_nodes", analyze_query_and_nodes)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("check_query", check_query)
    workflow.add_node("execute_query", execute_query)

    workflow.set_entry_point("analyze_query_and_nodes")

    workflow.add_edge("analyze_query_and_nodes", "generate_query")
    workflow.add_edge("generate_query", "check_query")
    workflow.add_conditional_edges(
        "check_query",
        decide_to_execute,
        {"execute": "execute_query", END: END},
    )
    workflow.add_edge("execute_query", END)

    return workflow.compile()

# Main execution block
if __name__ == "__main__":
    # This placeholder prompt for the analyzer was working before, so we'll keep it for the runnable script
    prompt_analyzer = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an intelligent analysis engine... Respond with a JSON object: {{'start_node': '...', 'end_node': '...'}}."),
            ("human", "Query: {query}\nCandidate Nodes: {nodes}"),
        ]
    )
    analyzer_runnable = prompt_analyzer | llm.with_structured_output(AnalysisResult)

    app = build_agent()
    print("LangGraph agent built successfully.")

    user_input = {
        "query": "流浪地球和刹车时代之间是什么关系，并且要求这个关系是关于'事件描述'的",
        "nodes": ["流浪地球", "刹车时代", "潮汐"],
        "relationships": {
            "keywords": "事件描述"
        }
    }
    print(f"\n--- Running Agent for input: {user_input} ---")

    result = app.invoke(user_input)
    print("\n---FINAL RESULT---")
    print(f"Input: {user_input}")
    print(f"Final Cypher Query: {result.get('cypher_query')}")
    if result.get('query_error'):
        print(f"Query Execution Error: {result.get('query_error')}")
    else:
        print(f"Query Result: {result.get('query_result')}")