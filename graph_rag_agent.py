import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from neo4j_utils import Neo4jCRUD

# Initialize Neo4j CRUD operations
# Replace with your Neo4j credentials
neo4j_crud = Neo4jCRUD("bolt://localhost:7687", "neo4j", "password")

def create_person_node(name: str) -> str:
    """Creates a person node in the graph."""
    neo4j_crud.create_node("Person", {"name": name})
    return f"Person node '{name}' created."

def create_relationship(person1_name: str, relationship_type: str, person2_name: str) -> str:
    """Creates a relationship between two person nodes."""
    node1 = neo4j_crud.find_node("Person", "name", person1_name)
    node2 = neo4j_crud.find_node("Person", "name", person2_name)
    if node1 and node2:
        neo4j_crud.create_relationship(node1, relationship_type, node2)
        return f"Relationship '{person1_name} {relationship_type} {person2_name}' created."
    else:
        return "One or both persons not found."

def find_person(name: str) -> str:
    """Finds a person node in the graph."""
    node = neo4j_crud.find_node("Person", "name", name)
    if node:
        return dict(node)
    return f"Person '{name}' not found."

tools = [
    Tool(
        name="CreatePerson",
        func=create_person_node,
        description="Creates a person node in the graph. Input should be the name of the person.",
    ),
    Tool(
        name="CreateRelationship",
        func=create_relationship,
        description="Creates a relationship between two person nodes. Input should be a comma-separated string of person1_name, relationship_type, person2_name.",
    ),
    Tool(
        name="FindPerson",
        func=find_person,
        description="Finds a person node in the graph. Input should be the name of the person.",
    ),
]

# Define the prompt template
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Initialize the LLM
llm = Ollama(model="llama2")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
