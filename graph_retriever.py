import json
from typing import List, Dict, Any

def get_graph_data() -> List[List[Dict[str, Any]]]:
    """
    从JSON文件中加载图数据。
    """
    with open('data/graph_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def search_graph_db(question: str, graph_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    在图数据中搜索与问题相关的实体和关系。

    这是一个简化的模拟搜索，它检查问题中是否包含图谱中的实体ID。
    在实际应用中，这里应该是一个更复杂的图查询逻辑（例如，使用Cypher查询Neo4j）。
    """
    found_relations = []
    # 将问题转换为小写以便不区分大小写地进行匹配
    lower_question = question.lower()

    for path in graph_data:
        for relation_obj in path:
            start_node_id = relation_obj.get("start_node", {}).get("entity_id", "").lower()
            end_node_id = relation_obj.get("end_node", {}).get("entity_id", "").lower()

            # 如果问题中提到了关系的开始节点或结束节点，则认为相关
            if start_node_id in lower_question or end_node_id in lower_question:
                found_relations.append(relation_obj)

    return found_relations

def format_graph_results(results: List[Dict[str, Any]]) -> str:
    """
    将图数据库的检索结果格式化为人类可读的字符串。
    """
    if not results:
        return "在图知识库中没有找到相关信息。"

    formatted_string = "从图知识库中找到以下关系：\n"
    for res in results:
        start_node = res.get('start_node', {}).get('entity_id', '未知实体')
        end_node = res.get('end_node', {}).get('entity_id', '未知实体')
        relation_desc = res.get('relation', {}).get('description', '无描述').replace('<SEP>', '；')

        formatted_string += f"- **关系**: {start_node} -> {end_node}\n"
        formatted_string += f"  - **描述**: {relation_desc}\n"

    return formatted_string

if __name__ == '__main__':
    # 示例用法
    sample_data = get_graph_data()
    sample_question = "地球和木星之间发生了什么？"
    search_results = search_graph_db(sample_question, sample_data)
    formatted_output = format_graph_results(search_results)

    print(f"针对问题: '{sample_question}'\n")
    print(formatted_output)

    sample_question_2 = "太阳对地球有什么影响？"
    search_results_2 = search_graph_db(sample_question_2, sample_data)
    formatted_output_2 = format_graph_results(search_results_2)

    print(f"\n针对问题: '{sample_question_2}'\n")
    print(formatted_output_2)