import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

# Configure the Tongyi model
llm = ChatOpenAI(
    temperature=0.7,
    model="qwen3-max",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Define the state for the graph
class AgentState(TypedDict):
    document: str
    mind_map: str

# Define the nodes for the graph
def generate_mind_map(state):
    """
    Generates a mind map from the given document.
    """
    prompt = ChatPromptTemplate.from_template(
        "Generate a mind map in markdown format for the following document:\n\n{document}"
    )
    chain = prompt | llm | StrOutputParser()
    mind_map = chain.invoke({"document": state["document"]})
    return {"mind_map": mind_map}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("generate_mind_map", generate_mind_map)
workflow.set_entry_point("generate_mind_map")
workflow.add_edge("generate_mind_map", END)

# Compile the graph
app = workflow.compile()

# Function to run the agent
def run_agent(document: str):
    """
    Runs the agent to generate a mind map from a document.
    """
    inputs = {"document": document}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
        print("---")
    print(value["mind_map"])
    return value["mind_map"]

if __name__ == "__main__":
    document_snippet = """
    The solar system is the gravitationally bound system of the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, the dwarf planets and small Solar System bodies. Of the objects that orbit the Sun indirectly—the natural satellites—two are larger than the smallest planet, Mercury, and one more is almost its size.
    """
    run_agent(document_snippet)
