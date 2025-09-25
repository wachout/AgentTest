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


# 1. Define Graph Schema (Updated)
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

# 3.1 Define the State for the graph (Updated)
class AgentState(TypedDict):
    query: str
    nodes: List[str]
    keywords: List[str]
    cypher_query: str
    query_result: Optional[List[dict]]
    query_error: Optional[str]
    errors: List[str]
    refine_count: int

# 3.2 Define the Nodes of the graph

# Node 1: Analyze Question (Updated)
class Keywords(BaseModel):
    """Keywords extracted from the user's query."""
    keywords: List[str]

prompt_analyzer = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert at extracting keywords from a user's query. "
         "Extract the main action-oriented or descriptive keywords. Do not extract entity names, as they are provided separately. "
         "Respond with a JSON object: {{'keywords': ['keyword1', 'keyword2']}}."
        ),
        ("human", "Extract keywords from this query: {query}"),
    ]
)
analyzer_runnable = prompt_analyzer | llm.with_structured_output(Keywords)

def analyze_question(state: AgentState):
    """Takes user-provided nodes and extracts keywords from the query."""
    print("---ANALYZING QUESTION & NODES---")
    # User-provided nodes are now the primary source of entities
    nodes = state["nodes"]
    query = state["query"]

    # Extract keywords from the query string
    analysis_result = analyzer_runnable.invoke({"query": query})

    print(f"User-provided Nodes: {nodes}, Extracted Keywords: {analysis_result.keywords}")
    return {
        "nodes": nodes,
        "keywords": analysis_result.keywords,
        "refine_count": 0,
        "errors": [],
    }

# Node 2: Generate Cypher Query (Updated prompt)
prompt_generate_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query generator. Your primary task is to write a Cypher query based on a list of user-provided node names and a natural language query. "
         "The query should be read-only (i.e., no CREATE, SET, DELETE). "
         "The nodes in the database have a property `entity_id` which holds their name. You MUST use `entity_id` for matching nodes.\n"
         "Graph Schema:\n{schema}\n"
         "---"
         "Instructions:\n"
         "1. **CRITICAL RULE on Variable Names**: Any query path must use `start_node` for the starting node and `end_node` for the ending node. Do not use single-character variables like `a`, `b`, or `n`.\n"
         "2. **CRITICAL RULE on Return Values**: The `RETURN` clause of the query MUST ALWAYS return the full node object(s) to include all their properties (e.g., `RETURN start_node`, `RETURN start_node, r, end_node`).\n"
         "3. Use the provided 'nodes' list to identify the primary entities. Match them using `WHERE start_node.entity_id IN [...]`.\n"
         "4. Use the 'keywords' to understand the user's intent (e.g., find relationships, get descriptions).\n"
         "---"
         "Example 1: User wants to find the relationship between 'Tidal Force' and 'Earth Engine'.\n"
         "Nodes: ['Tidal Force', 'Earth Engine']\n"
         "Query: 'What is the connection between them?'\n"
         "Generated Cypher: MATCH (start_node)-[r]-(end_node) WHERE start_node.entity_id = 'Tidal Force' AND end_node.entity_id = 'Earth Engine' RETURN start_node, r, end_node\n"
         "---"
         "Example 2: User wants to know more about 'Helium Flash'.\n"
         "Nodes: ['Helium Flash']\n"
         "Query: 'Tell me about Helium Flash'\n"
         "Generated Cypher: MATCH (start_node) WHERE start_node.entity_id = 'Helium Flash' RETURN start_node"
        ),
        ("human",
         "Generate a Cypher query.\n"
         "Query: {query}\n"
         "Nodes: {nodes}\n"
         "Keywords: {keywords}"
        ),
    ]
)
query_generator_runnable = prompt_generate_query | llm

def generate_query(state: AgentState):
    print("---GENERATING CYPHER QUERY---")
    generation = query_generator_runnable.invoke({
        "schema": graph_schema,
        "query": state["query"],
        "nodes": state["nodes"],
        "keywords": state["keywords"]
    })
    cypher_query = generation.content.strip()
    if "```cypher" in cypher_query:
        cypher_query = cypher_query.split("```cypher")[1].split("```")[0].strip()
    elif "```" in cypher_query:
        cypher_query = cypher_query.split("```")[1].strip()
    print(f"Generated Query: {cypher_query}")
    return {"cypher_query": cypher_query}

# Node 3: Check Cypher Query
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
        ("human", "Please check the following Cypher query.\nQuestion: {query}\nQuery: {cypher_query}"),
    ]
)
query_checker_runnable = prompt_check_query | llm.with_structured_output(QueryCheck)

def check_query(state: AgentState):
    print("---CHECKING CYPHER QUERY---")
    check_result = query_checker_runnable.invoke({"schema": graph_schema, "query": state["query"], "cypher_query": state["cypher_query"]})
    if check_result.is_correct:
        print("Query Check: CORRECT")
        return {"errors": []}
    else:
        print(f"Query Check: INCORRECT. Errors: {check_result.errors}")
        return {"errors": check_result.errors}

# Node 4: Refine Cypher Query
prompt_refine_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query refiner. Your task is to correct a Cypher query based on the provided error feedback. "
         "Pay close attention to the schema and the user's original query to ensure the refined query is valid and answers the question.\n"
         "Remember to use `entity_id` to match nodes by name.\n"
         "Schema:\n{schema}"
        ),
        ("human",
         "Please refine the following Cypher query based on the errors provided.\n"
         "Original Query: {query}\n"
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
        "query": state["query"],
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


# Node 5: Execute Query
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

# 3.4 Assemble the graph
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("check_query", check_query)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("execute_query", execute_query)

    workflow.set_entry_point("analyze_question")

    workflow.add_edge("analyze_question", "generate_query")
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

    # Updated example usage
    user_input = {
        "query": "潮汐和地球发动机之间有什么关系？",
        "nodes": ["潮汐", "地球发动机"]
    }
    print(f"\n--- Running Agent for input: {user_input} ---")

    result = app.invoke(user_input)
    print("\n---FINAL RESULT---")
    print(f"Query: {user_input['query']}")
    print(f"Nodes: {user_input['nodes']}")
    print(f"Final Cypher Query: {result.get('cypher_query')}")
    if result.get('query_error'):
        print(f"Query Execution Error: {result.get('query_error')}")
    else:
        print(f"Query Result: {result.get('query_result')}")