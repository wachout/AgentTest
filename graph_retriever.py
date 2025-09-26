import json
from typing import List, Dict, Any

def search_graph_db(question: str, graph_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    在传入的图数据中搜索与问题相关的实体和关系。

    这是一个简化的模拟搜索，它检查问题中是否包含图谱中的实体ID。
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