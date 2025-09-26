import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END

from graph_retriever import search_graph_db, format_graph_results

# --- 1. 环境与配置 ---
# 加载 .env 文件中的环境变量
load_dotenv()

# --- 2. LangGraph 状态定义 ---
# 定义工作流中各个节点之间传递的数据结构
class AgentState(TypedDict):
    query: str
    provider: str
    graph_data: List[List[Dict[str, Any]]]
    emb_data: List[str]
    text_results: str
    graph_results: str
    final_answer: str

# --- 3. 初始化 LLM 和 嵌入模型 ---
def get_llm(provider="deepseek"):
    """根据选择的提供商初始化并返回LLM实例"""
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "your_deepseek_api_key":
            raise ValueError("DeepSeek API密钥未设置。请在.env文件中设置DEEPSEEK_API_KEY。")
        return ChatOpenAI(temperature=0.6, model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com/v1")
    elif provider == "qwen":
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "your_aliyun_api_key":
            raise ValueError("阿里云API密钥未设置。请在.env文件中设置ALIYUN_API_KEY。")
        return ChatTongyi(temperature=0.7, model="qwen-long", api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}。请选择 'deepseek' 或 'qwen'。")

def get_embeddings_model():
    """初始化并返回嵌入模型"""
    try:
        return OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"无法初始化Ollama嵌入模型: {e}")
        print("请确保Ollama服务正在运行，并且已通过 `ollama pull nomic-embed-text` 下载了模型。")
        raise

# --- 4. LangGraph 节点定义 ---

def retrieve_text_node(state: AgentState) -> AgentState:
    """
    根据传入的文本数据动态创建FAISS索引并进行检索。
    """
    print("--- 节点: 动态文本检索 ---")
    query = state['query']
    emb_data = state['emb_data']

    if not emb_data:
        state['text_results'] = "文本知识库为空。"
        return state

    try:
        embeddings = get_embeddings_model()
        # 将传入的字符串列表转换为LangChain的Document对象
        documents = [Document(page_content=text) for text in emb_data]

        # 动态创建FAISS索引
        db = FAISS.from_documents(documents, embeddings)

        # 使用相似性搜索从向量数据库中检索文档
        retriever = db.as_retriever(search_kwargs={'k': 2}) # 检索最相关的2个文档块
        docs = retriever.invoke(query)

        formatted_results = "\n\n".join([doc.page_content for doc in docs])
        state['text_results'] = formatted_results
        print("动态文本检索完成。")
    except Exception as e:
        print(f"动态文本检索失败: {e}")
        state['text_results'] = f"文本检索过程中出错: {e}"

    return state

def retrieve_graph_node(state: AgentState) -> AgentState:
    """
    从传入的图数据中检索信息。
    """
    print("--- 节点: 动态图检索 ---")
    query = state['query']
    graph_data = state['graph_data']

    try:
        search_results = search_graph_db(query, graph_data)
        formatted_results = format_graph_results(search_results)
        state['graph_results'] = formatted_results
        print("动态图检索完成。")
    except Exception as e:
        print(f"动态图检索失败: {e}")
        state['graph_results'] = f"图检索过程中出错: {e}"

    return state

def generate_answer_node(state: AgentState) -> AgentState:
    """
    整合检索到的信息并使用LLM生成最终答案。
    """
    print("--- 节点: 生成答案 ---")
    query = state['query']
    provider = state['provider']
    text_context = state['text_results']
    graph_context = state['graph_results']

    prompt = f"""
你是一个智能问答助手。你的任务是根据下面提供的两个知识源，全面而准确地回答用户的问题。

**知识源1：文本知识库 (传入的文本段落)**
---
{text_context}
---

**知识源2：图知识库 (传入的图数据)**
---
{graph_context}
---

请综合以上两个知识源的信息，回答下面的问题。

用户问题是："{query}"
"""

    try:
        llm = get_llm(provider)
        response = llm.invoke([SystemMessage(content=prompt)])
        state['final_answer'] = response.content
        print("答案已生成。")
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        state['final_answer'] = f"生成答案时遇到错误: {e}"

    return state

# --- 5. 主函数：运行智能体 ---
def run_agent(param: Dict[str, Any]):
    """
    运行RAG智能体，处理动态传入的知识库。
    """
    # 构建并编译LangGraph工作流
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("retrieve_text", retrieve_text_node)
    graph_builder.add_node("retrieve_graph", retrieve_graph_node)
    graph_builder.add_node("generate_answer", generate_answer_node)

    graph_builder.set_entry_point("retrieve_text")
    graph_builder.add_edge('retrieve_text', 'retrieve_graph')
    graph_builder.add_edge('retrieve_graph', 'generate_answer')
    graph_builder.add_edge('generate_answer', END)

    app = graph_builder.compile()

    # 准备输入状态
    initial_state = {
        "query": param.get("query"),
        "provider": param.get("provider", "deepseek"), # 默认为deepseek
        "emb_data": param.get("emb_data", []),
        "graph_data": param.get("graph_data", [])
    }

    # 运行工作流
    try:
        final_state = app.invoke(initial_state)
        return final_state.get('final_answer', '未能生成最终答案。')
    except Exception as e:
        return f"执行智能体时出错: {e}"

# --- 6. 示例用法 ---
if __name__ == "__main__":
    # 使用用户提供的示例数据结构
    param = {
        "query": "根据《流浪地球》的背景，太阳和地球之间发生了什么？",
        "provider": "deepseek", # 或 "qwen"
        "graph_data": [
            [{"end_node": {"entity_id": "地球"}, "relation": {"description": "地球因太阳即将爆炸而需要逃离太阳系。<SEP>地球为了躲避太阳的威胁而进行逃生计划。"}, "start_node": {"entity_id": "太阳"}}],
            [{"end_node": {"entity_id": "地球"}, "relation": {"description": "当地球靠近木星时，木星的巨大引力几乎将地球捕获。"}, "start_node": {"entity_id": "木星"}}]
        ],
        "emb_data": [
            "在不久的将来，科学家们发现太阳正急速衰老、膨胀，即将发生一场名为“氦闪”的剧烈爆炸，整个太阳系都将被吞噬。",
            "为了自救，人类社会倾尽全球之力，启动了史无前例的“流浪地球”计划，将地球整个推离太阳系。",
            "当地球借助木星的引力进行加速变轨时，由于木星引力过强，地球被意外捕获，险些坠入木星大气层。"
        ]
    }

    print("--- 正在运行RAG智能体 ---")
    # 注意：运行此示例前，请确保Ollama服务正在运行，并且.env文件中已配置好API密钥。
    final_answer = run_agent(param)

    print("\n--- 最终答案 ---\n")
    print(final_answer)