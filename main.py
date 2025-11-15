import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, AsyncGenerator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up the LLM
# You can switch between DeepSeek and Tongyi by commenting/uncommenting the following lines
# DeepSeek
llm = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# Tongyi (Alibaba)
# llm = ChatTongyi(
#     temperature=0.7,
#     model="qwen3-32b",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# llm = llm.bind(enable_thinking=False)


# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]

# Define the nodes for our graph
def call_model(state: AgentState):
    """Calls the LLM to generate a response."""
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compile the graph
app_runnable = workflow.compile()

async def get_streaming_response(message: str, history: List[dict]) -> AsyncGenerator[str, None]:
    """
    This function takes a user message and chat history, and yields the streaming response from the agent.
    """
    history_messages = [AIMessage(content=msg['content']) if msg['role'] == 'assistant' else HumanMessage(content=msg['content']) for msg in history]

    inputs = {
        "messages": [SystemMessage(content="You are a helpful assistant.")] + history_messages + [HumanMessage(content=message)]
    }

    async for event in app_runnable.astream_events(inputs, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content

async def main():
    """
    Main function to run the agent with an example.
    """
    print("Agent: Hello! How can I help you today?")

    # Example conversation
    history = []

    # First message
    user_message_1 = "Hello, what's the weather like in London?"
    print(f"User: {user_message_1}")

    response_1 = ""
    async for chunk in get_streaming_response(user_message_1, history):
        response_1 += chunk
        print(f"Agent chunk: {chunk}")

    history.append({"role": "user", "content": user_message_1})
    history.append({"role": "assistant", "content": response_1})

    print(f"\nFull Agent Response 1: {response_1}\n")

    # Follow-up message
    user_message_2 = "What about in Paris?"
    print(f"User: {user_message_2}")

    response_2 = ""
    async for chunk in get_streaming_response(user_message_2, history):
        response_2 += chunk
        print(f"Agent chunk: {chunk}")

    history.append({"role": "user", "content": user_message_2})
    history.append({"role": "assistant", "content": response_2})

    print(f"\nFull Agent Response 2: {response_2}\n")


if __name__ == "__main__":
    asyncio.run(main())
