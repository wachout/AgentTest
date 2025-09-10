from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# 从我们的查询逻辑模块导入相关函数和模拟类
from query import (
    hybrid_search,
    MockConnection,
    MockCollection,
    rerank_results # 导入rerank_results以在mock模式下使用
)

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Knowledge Graph Hybrid Search API",
    description="An API that combines vector search with knowledge graph context for advanced document retrieval.",
    version="1.0.0"
)

# --- 数据模型 ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    content: str | None
    score: float
    related_info: dict

# --- 全局变量 ---
# 在应用启动时加载模型，避免重复加载
# 在这个模拟版本中，我们仍然加载模型以模拟真实场景
model = None
mock_neo4j_conn = None
mock_milvus_collection = None

@app.on_event("startup")
def startup_event():
    """
    应用启动时执行的事件：加载模型和模拟连接
    """
    global model, mock_neo4j_conn, mock_milvus_collection

    print("--- Server is starting up ---")
    print("Loading sentence transformer model...")
    # 在真实应用中，模型加载可能会花费一些时间
    model = SentenceTransformer('bge-small-zh-v1.5')
    print("Model loaded.")

    # 初始化模拟的数据库连接
    mock_neo4j_conn = MockConnection()
    mock_milvus_collection = MockCollection()
    print("Mock database connections initialized.")
    print("--- Server startup complete ---")


@app.post("/search", response_model=list[SearchResult])
def search(search_query: SearchQuery):
    """
    执行混合搜索的核心API端点。

    - **query**: The user's search query string.
    - **top_k**: The number of results to retrieve from the initial vector search.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded yet.")

    print(f"\nReceived query: '{search_query.query}' with top_k={search_query.top_k}")

    # --- 使用模拟数据执行混合搜索 ---
    # 在真实环境中，你会调用 hybrid_search(query, model, real_milvus_collection, real_neo4j_conn)
    # 这里我们为了演示，手动调用 rerank_results，因为它包含了核心的排序逻辑

    # 1. 模拟向量搜索返回的结果
    # 这里可以根据查询动态生成一些模拟结果，但为简单起见，我们使用固定的模拟数据
    mock_vector_results = [
        ("第二条", 0.35),
        ("第一条", 0.88)
    ]

    # 2. 模拟图数据库返回的上下文
    mock_graph_context = {
        "第一条": {
            "content": "第一条 为了加强和规范社会保险费征缴工作...",
            "entities": ["社会保险费", "社会保险金", "条例"],
            "keywords": ["社会保险费", "征缴工作", "社会保险金", "条例"],
            "chapter_title": "第一章 总 则"
        },
        "第二条": {
            "content": "第二条 基本养老保险费、基本医疗保险费...",
            "entities": ["社会保险费", "条例", "缴费单位", "缴费个人"],
            "keywords": ["社会保险费", "征收", "缴纳", "条例", "缴费单位", "缴费个人"],
            "chapter_title": "第一章 总 则"
        }
    }

    # 3. 执行重排序
    final_results = rerank_results(mock_vector_results, mock_graph_context, search_query.query)

    # 4. 格式化为API响应
    response = [SearchResult(**res) for res in final_results]

    return response

@app.get("/", summary="API Root", description="A simple hello world endpoint to check if the API is running.")
def read_root():
    return {"message": "Welcome to the Hybrid Search API. Use the /docs endpoint to see the API documentation."}

if __name__ == "__main__":
    """
    使用 uvicorn 启动 API 服务。
    在命令行中运行: uvicorn main:app --reload
    """
    print("--- To run this API, use the command: ---")
    print("uvicorn main:app --host 0.0.0.0 --port 8000 --reload")

    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # 直接运行此脚本不会启动服务器，请使用上面的命令行指令。
