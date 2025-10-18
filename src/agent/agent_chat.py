import os
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import operator

# Set your API keys and base URLs as environment variables
# For example:
# os.environ["DEEPSEEK_API_KEY"] = "your_deepseek_api_key"
# os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"
# os.environ["QWEN_API_KEY"] = "your_qwen_api_key"
# os.environ["QWEN_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- LLM Configuration ---
# It's recommended to use environment variables for API keys and base URLs
# For demonstration, we use placeholders. Replace them with your actual credentials or set env vars.
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-afce12abb3b142c787e85f8ec97c1ad2")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

qwen_api_key = os.getenv("QWEN_API_KEY", "sk-0270be722a48439e9ed73001e8e2524b")
qwen_base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Initialize the DeepSeek LLM
llm_deepseek = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner",
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
)

# Initialize the Qwen LLM (alternative)
# llm_qwen = ChatOpenAI(
#     temperature=0.7,
#     model="qwen-plus", # or another suitable model from dashscope
#     api_key=qwen_api_key,
#     base_url=qwen_base_url,
# )
# llm_qwen = llm_qwen.bind(enable_thinking=False)

# We will use the DeepSeek LLM by default
llm = llm_deepseek

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our RAG agent.
    """
    query: str
    graph_data: List[List[Dict[str, Any]]]
    emb_data: List[Dict[str, Any]]
    context: str
    answer: str

# --- Node Functions ---

def format_context(state: AgentState) -> Dict[str, Any]:
    """
    Formats the raw graph and embedding data into a unified context string.
    """
    graph_data = state["graph_data"]
    emb_data = state["emb_data"]

    context_parts = []

    # Format embedding data (Knowledge 1)
    if emb_data:
        context_parts.append("--- Retrieved Text Passages ---")
        for item in emb_data:
            context_parts.append(f"Title: {item.get('title', 'N/A')}\nContent: {item.get('content', 'N/A')}")
        context_parts.append("-" * 20)

    # Format graph data (Knowledge 2)
    if graph_data:
        context_parts.append("\n--- Retrieved Knowledge Graph Data ---")
        for path in graph_data:
            for relation_info in path:
                start_node = relation_info.get('start_node', {})
                end_node = relation_info.get('end_node', {})
                relation = relation_info.get('relation', {})

                context_parts.append(
                    f"({start_node.get('entity_id', 'Unknown')}) "
                    f"-[{relation.get('description', 'related to')}]-> "
                    f"({end_node.get('entity_id', 'Unknown')})"
                )
                context_parts.append(f"  - Start Node: {start_node.get('entity_id')}, Description: {start_node.get('description')}")
                context_parts.append(f"  - End Node: {end_node.get('entity_id')}, Description: {end_node.get('description')}")

    context = "\n\n".join(context_parts)
    return {"context": context}

def generate_answer(state: AgentState) -> Dict[str, Any]:
    """
    Generates an answer based on the user's query and the formatted context.
    """
    query = state["query"]
    context = state["context"]

    prompt = f"""
Based on the following context, please provide a comprehensive answer to the user's question.
If the context contains tables or image URLs relevant to the question, please include them in your answer.

--- Context ---
{context}
--- End of Context ---

User Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("format_context", format_context)
workflow.add_node("generate_answer", generate_answer)

# Define the workflow edges
workflow.set_entry_point("format_context")
workflow.add_edge("format_context", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the graph
app = workflow.compile()

# --- Main Execution Function ---
def run_agent(param: Dict[str, Any]) -> str:
    """
    Runs the RAG agent with the given parameters.
    """
    inputs = {
        "query": param["query"],
        "graph_data": param["graph_data"],
        "emb_data": param["emb_data"],
    }

    final_state = app.invoke(inputs)
    return final_state["answer"]

# --- Example Usage ---
if __name__ == "__main__":
    # Example parameters as provided in the problem description
    example_param = {
        "query": "太阳和地球有什么关系？",
        "graph_data": [
            [
                {
                    "start_node": {
                        "entity_id": "太阳",
                        "description": "太阳是地球绕行的恒星...",
                        "chunks": ["chunk1"],
                        "titles": ["title1"]
                    },
                    "end_node": {
                        "entity_id": "地球",
                        "description": "地球是人类居住的星球...",
                        "chunks": ["chunk1"],
                        "titles": ["title1"]
                    },
                    "relation": {
                        "description": "地球在变轨加速过程中，距离太阳的变化导致温度波动...",
                        "weight": 6.0
                    }
                }
            ]
        ],
        "emb_data": [
            {"score": 1.0, "title": "《流浪地球》", "content": "太阳是地球绕行的恒星，同时也是地球上生命的主要能源来源。"},
            {"score": 1.0, "title": "《流浪地球》", "content": "地球是人类居住的星球，也是故事发生的主要背景。"}
        ]
    }

    # Run the agent and print the answer
    answer = run_agent(example_param)
    print("--- Generated Answer ---")
    print(answer)

    # To run this script from the root directory:
    # python -m src.agent.agent_chat