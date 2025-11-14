import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, AsyncGenerator
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import json

# Load environment variables
load_dotenv()

# LLM Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Initialize the DeepSeek LLM
llm = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# Optional: Initialize the Tongyi LLM
# llm = ChatTongyi(
#     temperature=0.7,
#     model="qwen3-32b",
#     api_key=DASHSCOPE_API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# llm = llm.bind(enable_thinking=False)


# Define the state for the graph
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

# Define the agent node
async def call_model(state):
    messages = state['messages']
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

# Define the graph
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

# Compile the graph
graph = builder.compile()

# Initialize FastAPI app
app = FastAPI()

class Request(BaseModel):
    messages: List[dict]

async def stream_events(messages: List[AnyMessage]) -> AsyncGenerator[str, None]:
    """
    Streams events from the LangGraph agent.
    """
    async for output in graph.astream({"messages": messages}):
        for key, value in output.items():
            if key == "agent":
                yield f"data: {json.dumps(value['messages'][-1].content)}\n\n"

@app.post("/stream")
async def stream(request: Request):
    """
    API endpoint to stream the agent's response.
    """
    messages = [HumanMessage(content=m['content']) if m['role'] == 'user'
                else SystemMessage(content=m['content'])
                for m in request.messages]

    return StreamingResponse(stream_events(messages), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
