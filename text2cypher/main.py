import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
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
Node properties:
- entity_id: string (The name of the node, e.g., '潮汐')
- source_id: string (The ID of the source paragraph, e.g., 'chunk-...')
- description: string (A description of the node's context)
- entity_type: string (The type of the entity, e.g., 'UNKNOWN', 'Technology')
- file_path: string (The path to the source file)

Edge properties:
- relationship: string (The type of relationship between nodes)

Relationships connect nodes. For example:
(n1)-[:RELATED_TO]->(n2)
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

# 3.1 Define the State for the graph (Simplified)
class AgentState(TypedDict):
    nodes: List[str]
    cypher_query: str
    query_result: Optional[List[dict]]
    query_error: Optional[str]
    errors: List[str]
    refine_count: int

# 3.2 Define the Nodes of the graph

# Node 1: Generate Cypher Query (Simplified and now the entry point)
prompt_generate_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query generator. Your only task is to write a query that finds all nodes directly connected to a given list of nodes. "
         "The query must be read-only (i.e., no CREATE, SET, DELETE).\n"
         "Graph Schema:\n{schema}\n"
         "---"
         "Instructions:\n"
         "1. **CRITICAL RULE on Variable Names**: The query path must use `start_node` for the nodes from the input list and `end_node` for the connected nodes.\n"
         "2. **CRITICAL RULE on Return Values**: The `RETURN` clause MUST return the full `start_node` and `end_node` objects to include all their properties.\n"
         "3. Match the input nodes using their `entity_id` property with the `IN` operator (e.g., `WHERE start_node.entity_id IN $nodes`).\n"
         "---"
         "Example:\n"
         "Nodes: ['Tidal Force', 'Earth Engine']\n"
         "Generated Cypher: MATCH (start_node)-[r]-(end_node) WHERE start_node.entity_id IN $nodes RETURN start_node, r, end_node"
        ),
        ("human",
         "Generate a Cypher query for the following nodes: {nodes}"
        ),
    ]
)
query_generator_runnable = prompt_generate_query | llm

def generate_query(state: AgentState):
    """Generates a Cypher query to find neighbors of the input nodes."""
    print("---GENERATING CYPHER QUERY---")

    # The LangChain Neo4j driver passes parameters differently, so we'll just format it into the prompt for the LLM
    # but the actual execution would need parameters. For generation, this is fine.
    generation = query_generator_runnable.invoke({
        "schema": graph_schema,
        "nodes": state["nodes"]
    })

    cypher_query = generation.content.strip()
    # A more robust way to handle parameters for actual execution
    # This generated query is for inspection, the one for execution should use parameters.
    # For now, we generate a query that is directly runnable for simple cases.
    # A production system would use `session.run(query, nodes=state['nodes'])`
    # Let's generate a directly runnable query for simplicity here.
    nodes_list_str = ", ".join([f"'{node}'" for node in state["nodes"]])
    cypher_query = f"MATCH (start_node)-[r]-(end_node) WHERE start_node.entity_id IN [{nodes_list_str}] RETURN start_node, r, end_node"

    print(f"Generated Query: {cypher_query}")
    return {"cypher_query": cypher_query, "refine_count": 0, "errors": []}

# Node 2: Check Cypher Query
class QueryCheck(BaseModel):
    is_correct: bool
    errors: List[str]

prompt_check_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query checker... "
         "Respond with a JSON object: {{'is_correct': true, 'errors': []}} or {{'is_correct': false, 'errors': ['error message']}}.\n"
         "Schema:\n{schema}"
        ),
        ("human", "Please check the following Cypher query: {cypher_query}"),
    ]
)
query_checker_runnable = prompt_check_query | llm.with_structured_output(QueryCheck)

def check_query(state: AgentState):
    print("---CHECKING CYPHER QUERY---")
    check_result = query_checker_runnable.invoke({"schema": graph_schema, "cypher_query": state["cypher_query"]})
    if check_result.is_correct:
        print("Query Check: CORRECT")
        return {"errors": []}
    else:
        print(f"Query Check: INCORRECT. Errors: {check_result.errors}")
        return {"errors": check_result.errors}

# Node 3: Refine Cypher Query
prompt_refine_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query refiner. Your task is to correct a Cypher query based on the provided error feedback. "
         "The query should find all nodes directly connected to the input nodes. "
         "Remember to use `start_node` and `end_node` as variable names and return the full node objects.\n"
         "Schema:\n{schema}"
        ),
        ("human",
         "Please refine the following Cypher query based on the errors provided.\n"
         "Nodes: {nodes}\n"
         "Original Cypher: {cypher_query}\n"
         "Errors: {errors}"
        ),
    ]
)
query_refiner_runnable = prompt_refine_query | llm

def refine_query(state: AgentState):
    """Refines the Cypher query based on feedback."""
    print("---REFINING CYPHER QUERY---")
    refine_count = state.get("refine_count", 0) + 1

    generation = query_refiner_runnable.invoke({
        "schema": graph_schema,
        "nodes": state["nodes"],
        "cypher_query": state["cypher_query"],
        "errors": "\n".join(state["errors"]),
    })

    refined_query = generation.content.strip()
    if "```cypher" in refined_query:
        refined_query = refined_query.split("```cypher")[1].split("```")[0].strip()
    elif "```" in refined_query:
        refined_query = refined_query.split("```")[1].strip()

    print(f"Refined Query: {refined_query}")
    return {"cypher_query": refined_query, "refine_count": refine_count}


# Node 4: Execute Query
def execute_query(state: AgentState):
    print("---EXECUTING CYPHER QUERY---")
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
            # In a real scenario, you'd pass parameters to the query
            # result = session.run(state["cypher_query"], nodes=state["nodes"])
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


# 3.3 Define the Edges of the graph
def decide_to_refine_or_execute(state: AgentState):
    if not state.get("errors"):
        return "execute"
    else:
        return "refine" if state.get("refine_count", 0) < 2 else "end"

# 3.4 Assemble the graph (Simplified)
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("check_query", check_query)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("execute_query", execute_query)

    workflow.set_entry_point("generate_query") # New entry point

    workflow.add_edge("generate_query", "check_query")
    workflow.add_conditional_edges(
        "check_query",
        decide_to_refine_or_execute,
        {"refine": "refine_query", "execute": "execute_query", "end": END},
    )
    workflow.add_edge("refine_query", "check_query")
    workflow.add_edge("execute_query", END)

    return workflow.compile()

# Main execution block
if __name__ == "__main__":
    app = build_agent()
    print("LangGraph agent built successfully.")

    # Updated example usage (Simplified)
    user_input = {
        "nodes": ["潮汐", "地球发动机"]
    }
    print(f"\n--- Running Agent for input: {user_input} ---")

    result = app.invoke(user_input)
    print("\n---FINAL RESULT---")
    print(f"Nodes: {user_input['nodes']}")
    print(f"Final Cypher Query: {result.get('cypher_query')}")
    if result.get('query_error'):
        print(f"Query Execution Error: {result.get('query_error')}")
    else:
        print(f"Query Result: {result.get('query_result')}")