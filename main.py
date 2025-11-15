import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
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

# FastAPI app
app = FastAPI()

class RequestBody(TypedDict):
    message: str
    history: List[dict]

@app.post("/stream")
async def stream(body: RequestBody):
    """
    This endpoint streams the response from the agent.
    """
    history = [AIMessage(content=msg['content']) if msg['role'] == 'assistant' else HumanMessage(content=msg['content']) for msg in body['history']]

    inputs = {
        "messages": [SystemMessage(content="You are a helpful assistant.")] + history + [HumanMessage(content=body['message'])]
    }

    async def event_stream():
        async for event in app_runnable.astream_events(inputs, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
