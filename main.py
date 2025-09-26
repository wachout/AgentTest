import os
import argparse
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import StateGraph, END

from graph_retriever import get_graph_data, search_graph_db, format_graph_results

# --- 1. 环境与配置 ---
# 加载 .env 文件中的环境变量
load_dotenv()

# FAISS 索引的路径
FAISS_INDEX_PATH = "faiss_index"

# --- 2. LangGraph 状态定义 ---
# 定义工作流中各个节点之间传递的数据结构
class AgentState(TypedDict):
    question: str  # 用户提出的原始问题
    text_results: str  # 从文本知识库检索到的结果
    graph_results: str # 从图知识库检索到的结果
    final_answer: str  # LLM生成的最终答案

# --- 3. 初始化 LLM 和 嵌入模型 ---
def get_llm(provider="deepseek"):
    """根据选择的提供商初始化并返回LLM实例"""
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "your_deepseek_api_key":
            raise ValueError("DeepSeek API密钥未设置。请在.env文件中设置DEEPSEEK_API_KEY。")
        return ChatOpenAI(
            temperature=0.6,
            model="deepseek-chat", # 使用 deepseek-chat 模型
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
    elif provider == "qwen":
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key or api_key == "your_aliyun_api_key":
            raise ValueError("阿里云API密钥未设置。请在.env文件中设置ALIYUN_API_KEY。")
        return ChatTongyi(
            temperature=0.7,
            model="qwen-long", # 使用 qwen-long 模型
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    else:
        raise ValueError("不支持的LLM提供商。请选择 'deepseek' 或 'qwen'。")

# 初始化嵌入模型 (用于加载FAISS索引)
# 同样，请确保Ollama服务正在运行
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
except Exception as e:
    print(f"无法初始化Ollama嵌入模型: {e}")
    embeddings = None

# --- 4. LangGraph 节点定义 ---

def retrieve_text_node(state: AgentState) -> AgentState:
    """
    从FAISS向量存储中检索与问题相关的文本。
    """
    print("--- 节点: 文本检索 ---")
    question = state['question']

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"警告: FAISS索引目录 '{FAISS_INDEX_PATH}' 不存在。跳过文本检索。")
        print("请运行 'python knowledge_base_preparer.py' 来创建索引。")
        state['text_results'] = "文本知识库不可用（索引未创建）。"
        return state

    if not embeddings:
        state['text_results'] = "文本知识库不可用（嵌入模型未初始化）。"
        return state

    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        # 使用相似性搜索从向量数据库中检索文档
        retriever = db.as_retriever(search_kwargs={'k': 2}) # 检索最相关的2个文档块
        docs = retriever.invoke(question)

        formatted_results = "\n\n".join([doc.page_content for doc in docs])
        state['text_results'] = formatted_results
        print("文本检索完成。")
    except Exception as e:
        print(f"文本检索失败: {e}")
        state['text_results'] = f"文本检索过程中出错: {e}"

    return state

def retrieve_graph_node(state: AgentState) -> AgentState:
    """
    从图数据库（模拟的JSON文件）中检索信息。
    """
    print("--- 节点: 图检索 ---")
    question = state['question']

    try:
        graph_data = get_graph_data()
        search_results = search_graph_db(question, graph_data)
        formatted_results = format_graph_results(search_results)

        state['graph_results'] = formatted_results
        print("图检索完成。")
    except Exception as e:
        print(f"图检索失败: {e}")
        state['graph_results'] = f"图检索过程中出错: {e}"

    return state

def generate_answer_node(state: AgentState) -> AgentState:
    """
    整合检索到的信息并使用LLM生成最终答案。
    """
    print("--- 节点: 生成答案 ---")
    question = state['question']
    text_context = state['text_results']
    graph_context = state['graph_results']

    # 构建系统提示
    prompt = f"""
你是一个智能问答助手。你的任务是根据下面提供的两个知识源，全面而准确地回答用户的问题。

**知识源1：文本知识库**
这是从相关文档中摘录的段落：
---
{text_context}
---

**知识源2：图知识库**
这是从图数据库中提取的实体关系信息：
---
{graph_context}
---

请综合以上两个知识源的信息，回答下面的问题。如果一个知识源的信息缺失或不可用，请主要依赖另一个。如果两个知识源都无法提供有效信息，请告知用户你无法找到答案。

用户问题是："{question}"
"""

    try:
        # 获取选择的LLM
        llm = get_llm(args.provider)

        # 调用LLM生成答案
        response = llm.invoke([SystemMessage(content=prompt)])
        state['final_answer'] = response.content
        print("答案已生成。")
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        state['final_answer'] = f"生成答案时遇到错误: {e}"

    return state


# --- 5. 构建 LangGraph 工作流 ---

# 初始化图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("retrieve_text", retrieve_text_node)
workflow.add_node("retrieve_graph", retrieve_graph_node)
workflow.add_node("generate_answer", generate_answer_node)

# 定义图的边
# 使用并行边，同时执行文本和图的检索
workflow.add_edge("retrieve_text", "generate_answer")
workflow.add_edge("retrieve_graph", "generate_answer")

# 设置入口点
# 我们需要一个方法来将两个并行检索的起点连接起来
# LangGraph中，一个简单的序列化入口可以做到这一点，但为了并行，我们可以在调用时一起触发
# 这里我们定义一个并行执行的起点
workflow.set_entry_point("retrieve_text")
# 在这里，我们通过将两个检索节点连接到生成节点来创建一个“扇入”结构
# LangGraph 会等待 retrieve_text 和 retrieve_graph 都完成后，再执行 generate_answer
# 但为了启动并行，我们需要修改入口点逻辑。一个简单的方法是创建一个虚拟的起始节点。
# 或者更直接地，我们可以创建一个并行执行的分支。

# 重新定义图结构
# 为清晰起见，我们构建一个顺序工作流：
# 1. 首先，从文本知识库检索信息。
# 2. 接着，从图知识库检索信息。
# 3. 最后，将所有信息汇总生成答案。
graph_builder = StateGraph(AgentState)
graph_builder.add_node("retrieve_text", retrieve_text_node)
graph_builder.add_node("retrieve_graph", retrieve_graph_node)
graph_builder.add_node("generate_answer", generate_answer_node)

# 定义工作流的边（执行顺序）
graph_builder.set_entry_point("retrieve_text")
graph_builder.add_edge('retrieve_text', 'retrieve_graph')
graph_builder.add_edge('retrieve_graph', 'generate_answer')
graph_builder.add_edge('generate_answer', END)

# 编译图，生成可执行的应用
app = graph_builder.compile()


# --- 6. 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一个结合文本和图知识库的问答智能体。")
    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=["deepseek", "qwen"],
        help="选择使用的大语言模型提供商 ( 'deepseek' 或 'qwen' )。"
    )
    args = parser.parse_args()

    print(f"正在使用 {args.provider} 模型进行自动化测试...")

    # --- 自动化测试 ---
    # 我们将硬编码一个测试问题，以避免在非交互式环境中出现EOFError
    test_question = "根据《流浪地球》的背景，太阳和地球之间发生了什么？"
    print(f"\n测试问题: {test_question}")

    # 运行工作流
    inputs = {"question": test_question}
    try:
        final_state = app.invoke(inputs)
        print("\n--- 最终答案 ---\n")
        print(final_state.get('final_answer', '未能生成答案。'))
    except Exception as e:
        print(f"\n--- 测试过程中捕获到错误 ---\n")
        print(e)

    # --- 原始的交互式代码（已注释掉） ---
    # print("智能体已启动。输入 'exit' 或 'quit' 来结束程序。")
    # while True:
    #     user_question = input("\n请输入你的问题: ")
    #     if user_question.lower() in ['exit', 'quit']:
    #         break
    #     inputs = {"question": user_question}
    #     final_state = app.invoke(inputs)
    #     print("\n--- 最终答案 ---\n")
    #     print(final_state.get('final_answer', '未能生成答案。'))