import os
import asyncio
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.base import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
import dashscope
from dashscope import Generation
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
tongyi_api_key = os.getenv("TONGYI_API_KEY")
if not tongyi_api_key:
    raise ValueError("TONGYI_API_KEY not found in environment variables. Please create a .env file and add your TONGYI_API_KEY.")

# Set the Dashscope API key
dashscope.api_key = tongyi_api_key

# Initialize embeddings
embeddings = DashScopeEmbeddings(
    dashscope_api_key=tongyi_api_key,
    model="text-embedding-v1"
)

async def llm_wrapper(prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs):
    """
    Asynchronous wrapper for the Tongyi LLM using the dashscope library directly.
    Constructs a message list from the system prompt, history, and current prompt.
    """
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    if history_messages:
        for msg in history_messages:
            if msg.get("role") == "user":
                messages.append({'role': 'user', 'content': msg.get("content")})
            elif msg.get("role") == "assistant":
                messages.append({'role': 'assistant', 'content': msg.get("content")})

    messages.append({'role': 'user', 'content': prompt})

    # Make the API call using dashscope
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        result_format='message',
        temperature=0.7,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        print(f"LLM Error Code: {response.code}")
        print(f"LLM Error Message: {response.message}")
        # Return an empty string or a specific error format that lightrag can handle
        return "{}" # Return empty JSON to avoid breaking the parsing logic

async def embed_wrapper(texts: list[str], **kwargs):
    return embeddings.embed_documents(texts, **kwargs)


async def main():
    """
    Main function to run the RAG pipeline.
    Initializes LightRAG, ingests documents, and performs a query.
    """
    # 1. Initialize LightRAG
    rag = LightRAG(
        llm_model_func=llm_wrapper,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            func=embed_wrapper
        ),
        working_dir="./lightrag_data"
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # 2. Data Ingestion
    try:
        with open("kb1.txt", "r", encoding="utf-8") as f:
            kb1_content = f.read()
        with open("kb2.txt", "r", encoding="utf-8") as f:
            kb2_content = f.read()
    except FileNotFoundError as e:
        print(f"Error: Knowledge base file not found. {e}")
        print("Please ensure kb1.txt and kb2.txt are in the same directory.")
        return

    await asyncio.gather(
        rag.ainsert(kb1_content, ids=["kb1"]),
        rag.ainsert(kb2_content, ids=["kb2"])
    )
    print("Successfully inserted knowledge base documents.")

    # 3. Query
    query = "流浪地球小说和电影有什么区别？"
    print(f"\nQuerying with: '{query}'")

    response = await rag.aquery(query, param=QueryParam(mode="mix"))

    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    # Ensure the working directory exists, but do not clear it automatically
    if not os.path.exists("./lightrag_data"):
        os.makedirs("./lightrag_data")

    asyncio.run(main())