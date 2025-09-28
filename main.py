import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.graph import StateGraph, END

from graph_retriever import search_graph_db, format_graph_results

# --- 1. 环境与配置 ---
# 加载 .env 文件中的环境变量
load_dotenv()

# --- 2. LangGraph 状态定义 ---
class AgentState(TypedDict):
    query: str
    provider: str
    graph_data: List[List[Dict[str, Any]]]
    emb_data: List[Dict[str, Any]]
    text_results: str
    graph_results: str
    final_answer: str

# --- 3. 初始化 LLM ---
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

# --- 4. LangGraph 节点定义 ---

def retrieve_text_node(state: AgentState) -> AgentState:
    """
    *** 已更新：根据预先计算好的score对文本进行排序和筛选。***
    """
    print("--- 节点: 基于Score的文本检索 ---")
    emb_data = state['emb_data']

    if not emb_data:
        state['text_results'] = "文本知识库为空。"
        return state

    try:
        # 按 'score' 字段对列表进行降序排序
        sorted_data = sorted(emb_data, key=lambda x: x.get('score', 0.0), reverse=True)

        # 选取前2个结果（或所有结果，如果总数小于2）
        top_results = sorted_data[:2]

        # 格式化输出
        formatted_results = []
        for item in top_results:
            title = item.get('title', '未知来源')
            content = item.get('content', '')
            formatted_results.append(f"来源:《{title}》 (相似度分数: {item.get('score', 'N/A')})\n内容: {content}")

        state['text_results'] = "\n\n".join(formatted_results)
        print("基于Score的文本检索完成。")
    except Exception as e:
        print(f"文本检索失败: {e}")
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

**知识源1：文本知识库 (已根据相似度分数排序)**
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
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("retrieve_text", retrieve_text_node)
    graph_builder.add_node("retrieve_graph", retrieve_graph_node)
    graph_builder.add_node("generate_answer", generate_answer_node)

    graph_builder.set_entry_point("retrieve_text")
    graph_builder.add_edge('retrieve_text', 'retrieve_graph')
    graph_builder.add_edge('retrieve_graph', 'generate_answer')
    graph_builder.add_edge('generate_answer', END)

    app = graph_builder.compile()

    initial_state = {
        "query": param.get("query"),
        "provider": param.get("provider", "deepseek"),
        "emb_data": param.get("emb_data", []),
        "graph_data": param.get("graph_data", [])
    }

    try:
        final_state = app.invoke(initial_state)
        return final_state.get('final_answer', '未能生成最终答案。')
    except Exception as e:
        return f"执行智能体时出错: {e}"

# --- 6. 示例用法 ---
if __name__ == "__main__":
    param = {
        "query": "太阳和地球之间发生了什么？",
        "provider": "deepseek",
        "graph_data": [
            [{"end_node": {"entity_id": "地球"}, "relation": {"description": "地球因太阳即将爆炸而需要逃离太阳系。<SEP>地球为了躲避太阳的威胁而进行逃生计划。"}, "start_node": {"entity_id": "太阳"}}]
        ],
        "emb_data": [
            {"score": 0.8, "title": "三体", "content": "三体世界围绕着三颗不规则运动的太阳运行，文明在恒纪元与乱纪元之间挣扎。"},
            {"score": 1.0, "title": "《流浪地球》", "content": "太阳是地球绕行的恒星，同时也是地球上生命的主要能源来源。"},
            {"score": 0.9, "title": "《流浪地球》", "content": "地球是人类居住的星球，也是故事发生的主要背景。"}
        ]
    }

    print("--- 正在运行RAG智能体 (基于预计算Score) ---")
    # 注意：运行此示例前，请确保.env文件中已配置好API密钥。Ollama不再是必需的。
    final_answer = run_agent(param)

    print("\n--- 最终答案 ---\n")
    print(final_answer)