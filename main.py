import os
import argparse
from knowledge_graph_maker import GraphMaker, Ontology
from knowledge_graph_maker.types import Document, LLMClient
from llm_clients import DeepSeekClient, TongyiClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="使用 LLM 从文本中提取知识图谱。")
    parser.add_argument(
        "--llm",
        type=str,
        choices=["deepseek", "tongyi"],
        default="deepseek",
        help="选择要使用的大语言模型: 'deepseek' 或 'tongyi'。",
    )
    args = parser.parse_args()

    # 1. 初始化 LLM 客户端
    llm_client: LLMClient
    if args.llm == "deepseek":
        print("使用 DeepSeek 模型...")
        # 确保 DEEPSEEK_API_KEY 已设置
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("错误: DEEPSEEK_API_KEY 环境变量未设置。")
        llm_client = DeepSeekClient(model="deepseek-chat")
    elif args.llm == "tongyi":
        print("使用通义千问模型...")
        # 确保 TONGYI_API_KEY 已设置
        if not os.getenv("TONGYI_API_KEY"):
            raise ValueError("错误: TONGYI_API_KEY 环境变量未设置。")
        llm_client = TongyiClient(model="qwen-plus")
    else:
        raise ValueError("无效的 LLM 选择。请使用 'deepseek' 或 'tongyi'。")

    # 2. 定义本体
    ontology = Ontology(
        labels=[
            {"Person": "人物的姓名，不带任何形容词。"},
            {"Object": "物品或概念，不加冠词'the'。"},
            {"Event": "涉及多个人物的事件。"},
            "Place",
            "Document",
            "Organisation",
            "Action",
            {"Miscellaneous": "任何无法用其他标签分类的重要概念。"},
        ],
        relationships=["任意两个实体之间的关系"],
    )

    # 3. 准备示例文本并分块
    text = """
    在霍格沃茨的魔法世界里，哈利·波特收到了由海格送来的录取通知书。
    他和他的朋友们，罗恩·韦斯莱和赫敏·格兰杰，一起在对角巷购买了他们的学习用品。
    在奥利凡德的商店里，哈利买下了一根冬青木魔杖，其中含有凤凰的羽毛。
    这根羽毛来自邓布利多的凤凰福克斯。
    他们的最终目标是击败试图通过魂器获得永生的黑魔王伏地魔。
    """

    # 简单的分块策略：按句号分割
    chunks = [chunk.strip() for chunk in text.split("。") if chunk.strip()]
    documents = [Document(text=chunk, metadata={"source": "harry_potter_intro"}) for chunk in chunks]

    # 4. 初始化 GraphMaker 并生成图谱
    graph_maker = GraphMaker(ontology=ontology, llm_client=llm_client, verbose=True)
    edges = graph_maker.from_documents(documents)

    # 5. 打印结果
    print(f"\n--- 从文本中提取了 {len(edges)} 条关系 ---")
    for i, edge in enumerate(edges):
        print(f"\n关系 {i+1}:")
        print(f"  节点 1: {edge.node_1.name} ({edge.node_1.label})")
        print(f"  关系: {edge.relationship}")
        print(f"  节点 2: {edge.node_2.name} ({edge.node_2.label})")
        print(f"  来源: {edge.metadata.get('source', 'N/A')}")
        print(f"  原始文本块: \"{edge.metadata.get('original_text', 'N/A')}\"")

if __name__ == "__main__":
    main()