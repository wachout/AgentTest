import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser


# Load environment variables
load_dotenv()

# 1. Initialize LLMs
# DeepSeek LLM
deepseek_llm = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# Tongyi (DashScope) LLM
tongyi_llm = ChatTongyi(
    temperature=0.7,
    model_name="qwen2-72b-instruct",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)


# Choose the LLM to use for the agent
router_llm = deepseek_llm
tool_llm = tongyi_llm


# 2. Define the State for the graph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_node: str  # Field to store the routing decision

# 3. Define Nodes and Logic

# 3.1. Router Node Logic
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an expert at routing a user query to a tool.
You must classify the query into one of the following categories: 'file_deletion', 'database_query', or 'other'.
Return a JSON object with a single key 'route' and the value as the category."""),
        ("human", "{user_query}"),
    ]
)

router_chain = router_prompt | router_llm | JsonOutputParser()

def router_node(state: AgentState):
    """
    Determines the intent of the user's query and stores the result in the state.
    """
    print("---ROUTER---")
    user_message = state["messages"][-1].content
    route_result = router_chain.invoke({"user_query": user_message})
    route = route_result['route']
    print(f"Router output: {route}")
    # Return a dictionary to update the 'next_node' field in the state
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

    # Create a prompt to extract the filename in JSON format
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at extracting file names from user requests. Return a JSON object with a single key 'file_name' containing only the filename."),
            ("human", "Please extract the filename from the following text: {user_request}"),
        ]
    )

    # Create the extractor chain
    extractor = prompt | tool_llm | JsonOutputParser()

    extracted_data = extractor.invoke({"user_request": user_message})
    filename = extracted_data['file_name']

    # Simulate file deletion
    response_message = f"Confirmed: The file '{filename}' will be deleted."
    print(response_message)
    state["messages"].append(AIMessage(content=response_message))
    return {"messages": state["messages"]}

# 3.3. Database Query Node Logic
def database_query_node(state: AgentState):
    """
    Handles database query intent (placeholder).
    """
    print("---DATABASE QUERY---")
    response_message = "I understand you have a question about the database. How can I assist you with data analysis?"
    print(response_message)
    state["messages"].append(AIMessage(content=response_message))
    return {"messages": state["messages"]}

# 3.4. Chit-Chat Node Logic
def chitchat_node(state: AgentState):
    """
    Handles general conversation.
    """
    print("---CHIT-CHAT---")
    user_message = state["messages"][-1]

    # Use the tool_llm for conversation
    response = tool_llm.invoke([user_message])

    print(f"Chit-chat response: {response.content}")
    state["messages"].append(response)
    return {"messages": state["messages"]}


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


# 5. Automated Test Cases
if __name__ == "__main__":
    print("--- Running Automated Test Cases ---")

    # Test Case 1: File Deletion
    print("\n--- Test Case 1: File Deletion ---")
    inputs_1 = {"messages": [HumanMessage(content="I need to delete the file 'old_data.csv'")]}
    final_state_1 = app.invoke(inputs_1)
    agent_response_1 = final_state_1["messages"][-1].content
    print(f"User Input: \"I need to delete the file 'old_data.csv'\"")
    print(f"Agent Response: {agent_response_1}")

    # Test Case 2: Database Query
    print("\n--- Test Case 2: Database Query ---")
    inputs_2 = {"messages": [HumanMessage(content="What is the total revenue for the last quarter?")]}
    final_state_2 = app.invoke(inputs_2)
    agent_response_2 = final_state_2["messages"][-1].content
    print(f"User Input: \"What is the total revenue for the last quarter?\"")
    print(f"Agent Response: {agent_response_2}")

    # Test Case 3: Chit-Chat
    print("\n--- Test Case 3: Chit-Chat ---")
    inputs_3 = {"messages": [HumanMessage(content="Good morning! How are you?")]}
    final_state_3 = app.invoke(inputs_3)
    agent_response_3 = final_state_3["messages"][-1].content
    print(f"User Input: \"Good morning! How are you?\"")
    print(f"Agent Response: {agent_response_3}")

    print("\n--- Test Cases Finished ---")
