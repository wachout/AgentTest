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
- id: string (unique identifier for the node, also serves as the name)
- name: string (name of the node)
- type: string ('实体' or '关键词')

Edge properties:
- relationship: string (type of the relationship, e.g., 'RELATED_TO')

The relationships are:
(:实体)-[:RELATED_TO]->(:实体)
(:实体)-[:RELATED_TO]->(:关键词)
(:关键词)-[:RELATED_TO]->(:实体)
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

# 3.1 Define the State for the graph
class AgentState(TypedDict):
    question: str
    entities: List[str]
    keywords: List[str]
    cypher_query: str
    query_result: Optional[List[dict]]
    query_error: Optional[str]
    errors: List[str]
    refine_count: int

# 3.2 Define the Nodes of the graph

# Node 1: Analyze Question
class Entity(BaseModel):
    name: str
    type: str

class ExtractedData(BaseModel):
    entities: List[Entity]
    keywords: List[str]

prompt_analyzer = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert at extracting entities and keywords from a user's question. "
         "Respond with a JSON object that strictly follows this format: "
         '{{ "entities": [{{ "name": "entity_name", "type": "entity_type" }}], "keywords": ["keyword1", "keyword2"] }}.'
        ),
        ("human", "{question}"),
    ]
)
analyzer_runnable = prompt_analyzer | llm.with_structured_output(ExtractedData)

def analyze_question(state: AgentState):
    print("---ANALYZING QUESTION---")
    question = state["question"]
    analysis_result = analyzer_runnable.invoke({"question": question})
    entity_names = [entity.name for entity in analysis_result.entities]
    print(f"Entities: {entity_names}, Keywords: {analysis_result.keywords}")
    return {
        "entities": entity_names,
        "keywords": analysis_result.keywords,
        "refine_count": 0,
        "errors": [],
    }

# Node 2: Generate Cypher Query (with improved prompt)
prompt_generate_query = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an expert Neo4j Cypher query generator. Your task is to write a Cypher query based on the provided entities, keywords, and graph schema. "
         "The query should be read-only (i.e., no CREATE, SET, DELETE). "
         "Pay close attention to the schema to ensure the query is valid. The user's question is about finding relationships between entities and keywords.\n"
         "Schema:\n{schema}\n"
         "Example 1: Find entities related to 'Graph Database'.\n"
         "MATCH (n {{id: 'Graph Database'}})-[:RELATED_TO]->(related) RETURN related.id, related.name\n"
         "Example 2: What is Neo4j?\n"
         "MATCH (n {{id: 'Neo4j'}}) RETURN n.id, n.name"
        ),
        ("human",
         "Generate a Cypher query to answer the following question.\n"
         "Question: {question}\n"
         "Entities: {entities}\n"
         "Keywords: {keywords}"
        ),
    ]
)
query_generator_runnable = prompt_generate_query | llm

def generate_query(state: AgentState):
    print("---GENERATING CYPHER QUERY---")
    generation = query_generator_runnable.invoke({"schema": graph_schema, "question": state["question"], "entities": state["entities"], "keywords": state["keywords"]})
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
         "You are an expert Neo4j Cypher query checker. Your task is to evaluate a generated Cypher query for correctness. "
         "Check for syntax errors, logical errors, and adherence to the graph schema. "
         "Respond with a JSON object that strictly follows this format: "
         '{{ "is_correct": true, "errors": [] }} or {{ "is_correct": false, "errors": ["error message"] }}.\n'
         "Schema:\n{schema}"
        ),
        ("human", "Please check the following Cypher query.\nQuestion: {question}\nQuery: {query}"),
    ]
)
query_checker_runnable = prompt_check_query | llm.with_structured_output(QueryCheck)

def check_query(state: AgentState):
    print("---CHECKING CYPHER QUERY---")
    check_result = query_checker_runnable.invoke({"schema": graph_schema, "question": state["question"], "query": state["cypher_query"]})
    if check_result.is_correct:
        print("Query Check: CORRECT")
        return {"errors": []}
    else:
        print(f"Query Check: INCORRECT. Errors: {check_result.errors}")
        return {"errors": check_result.errors}

# Node 4: Refine Cypher Query
prompt_refine_query = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Neo4j Cypher query refiner... Schema:\n{schema}"),
        ("human", "Please refine the following Cypher query...\nQuestion: {question}\nOriginal Query: {query}\nErrors: {errors}"),
    ]
)
query_refiner_runnable = prompt_refine_query | llm

def refine_query(state: AgentState):
    print("---REFINING CYPHER QUERY---")
    refine_count = state.get("refine_count", 0) + 1
    generation = query_refiner_runnable.invoke({"schema": graph_schema, "question": state["question"], "query": state["cypher_query"], "errors": "\n".join(state["errors"])})
    refined_query = generation.content.strip()
    if "```cypher" in refined_query:
        refined_query = refined_query.split("```cypher")[1].split("```")[0].strip()
    elif "```" in refined_query:
        refined_query = refined_query.split("```")[1].strip()
    print(f"Refined Query: {refined_query}")
    return {"cypher_query": refined_query, "refine_count": refine_count}


# Node 5: Execute Query
def execute_query(state: AgentState):
    """Executes the Cypher query against the Neo4j database."""
    print("---EXECUTING CYPHER QUERY---")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]) or uri == "bolt://localhost:7687":
        error_msg = "Neo4j connection details are not configured in .env file or are default."
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
    """Decides the next step after checking the query."""
    if not state.get("errors"):
        print("---DECISION: EXECUTE QUERY---")
        return "execute"
    else:
        if state.get("refine_count", 0) >= 2:
            print("---DECISION: MAX REFINEMENTS REACHED, ENDING---")
            return "end"
        else:
            print("---DECISION: REFINE QUERY---")
            return "refine"

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
        {
            "refine": "refine_query",
            "execute": "execute_query",
            "end": END,
        },
    )
    workflow.add_edge("refine_query", "check_query")
    workflow.add_edge("execute_query", END)

    return workflow.compile()

# Main execution block
if __name__ == "__main__":
    app = build_agent()
    print("LangGraph agent built successfully.")

    question = "What is Neo4j?"
    print(f"\n--- Running Agent for question: '{question}' ---")
    inputs = {"question": question}

    result = app.invoke(inputs)
    print("\n---FINAL RESULT---")
    print(f"Question: {question}")
    print(f"Final Cypher Query: {result.get('cypher_query')}")
    if result.get('query_error'):
        print(f"Query Execution Error: {result.get('query_error')}")
    else:
        print(f"Query Result: {result.get('query_result')}")