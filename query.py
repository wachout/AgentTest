import json
from neo4j import GraphDatabase
from pymilvus import connections, utility, Collection
from sentence_transformers import SentenceTransformer

# --- 连接信息和配置 ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "legal_documents"

MODEL_NAME = 'bge-small-zh-v1.5'
TOP_K = 5 # 向量检索时返回的最相似结果数量

# --- 模拟数据库连接和模型加载 ---
# 在实际应用中，这些对象应该被妥善管理
class MockConnection:
    """一个模拟的数据库连接，用于在没有实际数据库的情况下开发"""
    def run_query(self, query, parameters=None):
        print(f"--- MOCK NEO4J QUERY ---\nQuery: {query}\nParams: {parameters}\n")
        # 返回一个模拟的、符合结构的空结果
        return []

class MockCollection:
    """一个模拟的Milvus集合"""
    def search(self, data, *args, **kwargs):
        print(f"--- MOCK MILVUS SEARCH ---\nTop K: {kwargs.get('limit')}\n")
        # 返回一个模拟的、符合结构的空结果
        class MockHits:
            def __init__(self):
                self.ids = []
                self.distances = []
        class MockResult:
            def __init__(self):
                self.hits = [MockHits()]
        return [MockResult()]
    def load(self):
        print("--- MOCK MILVUS: Loading collection ---")

# --- 核心查询功能 ---

def search_vector(query_text, model, collection, top_k=TOP_K):
    """
    在Milvus中执行向量相似度搜索
    """
    print(f"\nStep 1: Performing vector search for query: '{query_text}'")
    query_vector = model.encode([query_text])

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    # 在实际环境中，使用真实的 collection 对象
    results = collection.search(
        data=query_vector,
        anns_field="vector",
        param=search_params,
        limit=top_k,
        expr=None,
    )

    hits = results[0]
    # 假设 hits.ids 和 hits.distances 分别是ID和距离列表
    return list(zip(hits.ids, hits.distances)) if hits else []

def search_graph(conn, sub_chapter_titles):
    """
    在Neo4j中查询与召回的子章节相关的图谱信息
    """
    print(f"\nStep 2: Fetching graph context for titles: {sub_chapter_titles}")
    if not sub_chapter_titles:
        return {}

    # 这个查询会获取每个子章节的实体、关键词和其所属的父章节
    query = """
    UNWIND $titles AS sub_title
    MATCH (sc:SubChapter {title: sub_title})
    OPTIONAL MATCH (sc)-[:MENTIONS_ENTITY]->(e:Entity)
    OPTIONAL MATCH (sc)-[:HAS_KEYWORD]->(k:Keyword)
    OPTIONAL MATCH (c:Chapter)-[:HAS_SUB_CHAPTER]->(sc)
    RETURN sc.title AS title,
           sc.content AS content,
           collect(DISTINCT e.name) AS entities,
           collect(DISTINCT k.name) AS keywords,
           c.title AS chapter_title
    """

    results = conn.run_query(query, parameters={'titles': sub_chapter_titles})

    graph_context = {
        record['title']: {
            "content": record['content'],
            "entities": record['entities'],
            "keywords": record['keywords'],
            "chapter_title": record['chapter_title']
        } for record in results
    }
    return graph_context

def rerank_results(vector_results, graph_context, query_text):
    """
    结合向量得分和图谱信息对结果进行重排序
    """
    print("\nStep 3: Reranking results based on graph context.")
    final_results = []

    for doc_id, distance in vector_results:
        if doc_id not in graph_context:
            continue

        context = graph_context[doc_id]

        # 计算一个简单的增强分数
        # 策略：如果查询文本中的词语出现在关键词或实体中，则加分
        bonus_score = 0
        for keyword in context.get('keywords', []):
            if keyword in query_text:
                bonus_score += 1
        for entity in context.get('entities', []):
            if entity in query_text:
                bonus_score += 1

        # 综合得分：原始距离得分 + 奖励分 (可以根据需求设计更复杂的算法)
        # 注意：距离越小越好，所以我们用 1/distance
        initial_score = 1.0 / (1.0 + distance) if distance is not None else 0
        final_score = initial_score + bonus_score

        final_results.append({
            "id": doc_id,
            "content": context.get('content'),
            "score": final_score,
            "distance": distance,
            "graph_bonus": bonus_score,
            "related_info": {
                "chapter": context.get('chapter_title'),
                "entities": context.get('entities'),
                "keywords": context.get('keywords')
            }
        })

    # 按最终得分降序排序
    final_results.sort(key=lambda x: x['score'], reverse=True)
    return final_results


def hybrid_search(query_text, model, collection, conn):
    """
    执行混合搜索
    """
    # 1. 向量召回
    vector_results = search_vector(query_text, model, collection)
    if not vector_results:
        print("No results from vector search.")
        return []

    recalled_titles = [result[0] for result in vector_results]

    # 2. 图谱信息获取
    graph_context = search_graph(conn, recalled_titles)

    # 3. 重排序
    ranked_results = rerank_results(vector_results, graph_context, query_text)

    return ranked_results

if __name__ == "__main__":
    print("--- Running Hybrid Search (Mock Mode) ---")

    # --- 模拟环境 ---
    # 加载模型 (仅用于编码查询)
    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    # 模拟数据库连接
    mock_neo4j_conn = MockConnection()
    mock_milvus_collection = MockCollection()

    # --- 执行查询 ---
    user_query = "关于社会保险费的缴纳单位有什么规定？"

    # 模拟从向量搜索返回的结果
    # 在真实场景中，这将由 search_vector 函数返回
    mock_vector_results = [
        ("第二条", 0.35), # (doc_id, distance)
        ("第一条", 0.88)
    ]

    # 模拟从图数据库返回的上下文
    # 在真实场景中，这将由 search_graph 函数返回
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

    # 执行重排序逻辑
    final_results = rerank_results(mock_vector_results, mock_graph_context, user_query)

    # --- 打印结果 ---
    print("\n--- Final Ranked Results ---")
    for result in final_results:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nNOTE: This script runs in 'mock' mode and does not connect to live databases.")
    print("To run in 'live' mode, you would replace mock objects with real connections.")
