import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser


# Load environment variables from .env.example
load_dotenv(dotenv_path=".env.example")

# 1. Initialize LLMs
# DeepSeek LLM
deepseek_llm = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner",
    api_key=os.getenv("DEEPSEEK_API_KEY", "dummy_key"),
    base_url="https://api.deepseek.com/v1",
)

# Tongyi (DashScope) LLM
tongyi_llm = ChatTongyi(
    temperature=0.7,
    model_name="qwen2-72b-instruct",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", "dummy_key"),
)


# Choose the LLM to use for the agent
router_llm = deepseek_llm
tool_llm = tongyi_llm


# 2. Define the State for the graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_node: str

# Helper function to format chat history
def format_chat_history(messages: List[BaseMessage]):
    """Formats the chat history into a readable string."""
    return "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in messages)

# 3. Define Nodes and Logic

# 3.1. Router Node Logic
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an expert at routing a user query to a tool based on the conversation history.
You must classify the latest user query into one of the following categories: 'file_deletion', 'database_query', or 'other'.
Consider the full conversation history to understand the user's intent, especially for follow-up questions. For example, if a user mentions a report and then says "delete it", the intent is likely 'file_deletion'.
Return a JSON object with a single key 'route' and the value as the category.

Here is the conversation history:
{chat_history}"""),
        ("human", "{user_query}"),
    ]
)

router_chain = router_prompt | router_llm | JsonOutputParser()

def router_node(state: AgentState):
    """
    Determines the intent of the user's query and stores the result in the state.
    """
    print("---ROUTER---")
    messages = state["messages"]
    user_query = messages[-1].content
    history = format_chat_history(messages[:-1]) # All messages except the last one

    route_result = router_chain.invoke({"user_query": user_query, "chat_history": history})
    route = route_result['route']
    print(f"Router output: {route}")
    return {"next_node": route}

def should_continue(state: AgentState):
    """
    This function is used by the conditional edge to determine the next node.
    """
    return state["next_node"]


# 3.2. File Deletion Node Logic
def file_deletion_node(state: AgentState):
    """
    Extracts the filename from the user query and simulates deletion.
    """
    print("---FILE DELETION---")
    user_message = state["messages"][-1].content

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at extracting file names from user requests. Return a JSON object with a single key 'file_name' containing only the filename. If the user refers to a file with a pronoun (e.g., 'it', 'that'), use the conversation context to identify the file. For example, if the history mentions 'sales_report.csv' and the user says 'delete it', you should identify 'sales_report.csv'."),
            ("human", "Conversation History:\n{chat_history}\n\nPlease extract the filename from the following text: {user_request}"),
        ]
    )

    extractor = prompt | tool_llm | JsonOutputParser()

    history = format_chat_history(state["messages"][:-1])
    extracted_data = extractor.invoke({"user_request": user_message, "chat_history": history})
    filename = extracted_data['file_name']

    response_message = f"Confirmed: The file '{filename}' will be deleted."
    print(response_message)
    new_messages = state["messages"] + [AIMessage(content=response_message)]
    return {"messages": new_messages}

# 3.3. Database Query Node Logic
def database_query_node(state: AgentState):
    """
    Handles database query intent (placeholder).
    """
    print("---DATABASE QUERY---")
    response_message = "Here is the summary of last month's sales data, which is stored in 'sales_report.csv'."
    print(response_message)
    new_messages = state["messages"] + [AIMessage(content=response_message)]
    return {"messages": new_messages}

# 3.4. Chit-Chat Node Logic
def chitchat_node(state: AgentState):
    """
    Handles general conversation using the full message history.
    """
    print("---CHIT-CHAT---")

    response = tool_llm.invoke(state["messages"])

    print(f"Chit-chat response: {response.content}")
    new_messages = state["messages"] + [response]
    return {"messages": new_messages}


# 4. Construct the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("file_deletion_tool", file_deletion_node)
workflow.add_node("database_query_tool", database_query_node)
workflow.add_node("chitchat", chitchat_node)

# Set the entry point
workflow.set_entry_point("router")

# Add conditional edges from the router
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "file_deletion": "file_deletion_tool",
        "database_query": "database_query_tool",
        "other": "chitchat",
    },
)

# Add edges to the end
workflow.add_edge("file_deletion_tool", END)
workflow.add_edge("database_query_tool", END)
workflow.add_edge("chitchat", END)

# Compile the graph
app = workflow.compile()


# 5. Automated Multi-Turn Test Case
if __name__ == "__main__":
    print("--- Running Automated Multi-Turn Test Case ---")

    # Initial state
    initial_messages = [HumanMessage(content="Can you give me a summary of last month's sales data?")]

    # First turn
    print("\n--- Turn 1 ---")
    turn_1_state = app.invoke({"messages": initial_messages})
    print(f"User: \"{initial_messages[0].content}\"")
    print(f"Agent: \"{turn_1_state['messages'][-1].content}\"")

    # Second turn
    print("\n--- Turn 2 ---")
    # The new state includes the history from the first turn
    turn_2_messages = turn_1_state["messages"] + [HumanMessage(content="Great. Now, please delete it.")]
    turn_2_state = app.invoke({"messages": turn_2_messages})

    print(f"User: \"{turn_2_messages[-1].content}\"")
    print(f"Agent: \"{turn_2_state['messages'][-1].content}\"")

    print("\n--- Multi-Turn Test Finished ---")
