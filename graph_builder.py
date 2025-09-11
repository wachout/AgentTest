import os
from typing import TypedDict, List, Dict, Any
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# --- Import Prompts ---
from prompts import CHAPTER_SPLITTER_PROMPT, PARAGRAPH_SPLITTER_PROMPT, SEMANTIC_SPLITTER_PROMPT

# This will be the state of our graph.
class AgentState(TypedDict):
    input_text: str
    chapter_splits: List[int]
    paragraph_splits: List[int]
    semantic_splits: List[int]
    error: str

# This is the Pydantic model we expect the LLM to return.
class SplitIndices(BaseModel):
    indices: List[int] = Field(description="A list of starting character indices for each split.")

# This function will act as a router to dispatch tasks to the parallel agent nodes.
def route_to_splitters(state: AgentState) -> List[str]:
    print("--- Routing to all splitters for parallel execution ---")
    return ["chapter_splitter", "paragraph_splitter", "semantic_splitter"]

# --- Graph Building ---

def create_graph():
    """
    Creates and compiles the LangGraph for text analysis.
    This graph is configured to run the three splitter agents in parallel.
    """

    # --- LLM and Parser Configuration (Initialized inside the factory function) ---
    llm = ChatOpenAI(
        temperature=0.6,
        model="deepseek-coder",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    json_parser = JsonOutputParser(pydantic_object=SplitIndices)

    # --- Agent Node Implementations (Defined inside to have access to LLM and parser) ---
    # Each node now returns ONLY the fields it modifies.
    async def run_chapter_splitter(state: AgentState) -> Dict[str, Any]:
        print("--- Running Chapter Splitter ---")
        try:
            chain = CHAPTER_SPLITTER_PROMPT | llm | json_parser
            result = await chain.ainvoke({"input_text": state["input_text"]})
            return {"chapter_splits": result['indices']}
        except Exception as e:
            print(f"ERROR in Chapter Splitter: {e}")
            return {"error": "Failed during chapter splitting."}

    async def run_paragraph_splitter(state: AgentState) -> Dict[str, Any]:
        print("--- Running Paragraph Splitter ---")
        try:
            chain = PARAGRAPH_SPLITTER_PROMPT | llm | json_parser
            result = await chain.ainvoke({"input_text": state["input_text"]})
            return {"paragraph_splits": result['indices']}
        except Exception as e:
            print(f"ERROR in Paragraph Splitter: {e}")
            return {"error": "Failed during paragraph splitting."}

    async def run_semantic_splitter(state: AgentState) -> Dict[str, Any]:
        print("--- Running Semantic Splitter ---")
        try:
            chain = SEMANTIC_SPLITTER_PROMPT | llm | json_parser
            result = await chain.ainvoke({"input_text": state["input_text"]})
            return {"semantic_splits": result['indices']}
        except Exception as e:
            print(f"ERROR in Semantic Splitter: {e}")
            return {"error": "Failed during semantic splitting."}

    # --- Graph Wiring ---
    workflow = StateGraph(AgentState)

    workflow.add_node("chapter_splitter", run_chapter_splitter)
    workflow.add_node("paragraph_splitter", run_paragraph_splitter)
    workflow.add_node("semantic_splitter", run_semantic_splitter)
    workflow.add_node("start_node", lambda state: state)

    workflow.set_entry_point("start_node")
    workflow.add_conditional_edges("start_node", route_to_splitters)
    workflow.add_edge("chapter_splitter", END)
    workflow.add_edge("paragraph_splitter", END)
    workflow.add_edge("semantic_splitter", END)

    app = workflow.compile()
    print("LangGraph for parallel execution compiled successfully!")
    return app
