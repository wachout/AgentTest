import json
from langgraph.graph import StateGraph, START, END
from agents import (
    AgentState,
    preprocess_text,
    extract_title,
    extract_summary,
    extract_toc,
    extract_metadata,
    classify_doctype,
    extract_authors,
)

# --- 1. Define the Graph ---
# Create a new graph
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("preprocess", preprocess_text)
workflow.add_node("extract_title", extract_title)
workflow.add_node("extract_summary", extract_summary)
workflow.add_node("extract_toc", extract_toc)
workflow.add_node("extract_metadata", extract_metadata)
workflow.add_node("classify_doctype", classify_doctype)
workflow.add_node("extract_authors", extract_authors)

# --- 2. Define the Edges ---
# The graph starts with the preprocessing node
workflow.add_edge(START, "preprocess")

# After preprocessing, run all extraction nodes in parallel
workflow.add_edge("preprocess", "extract_title")
workflow.add_edge("preprocess", "extract_summary")
workflow.add_edge("preprocess", "extract_toc")
workflow.add_edge("preprocess", "extract_metadata")
workflow.add_edge("preprocess", "classify_doctype")
workflow.add_edge("preprocess", "extract_authors")

# After all parallel nodes are done, end the graph
# We need a joining node to wait for all parallel tasks to complete.
# Let's add a simple collector node.
def collector_node(state: AgentState) -> AgentState:
    """A simple node to act as a join point for parallel edges."""
    print("---(Node: Collector)---")
    # No-op, just collects the state from parallel branches.
    # The state is automatically merged by LangGraph.
    print("   All extraction tasks complete.")
    return {}

workflow.add_node("collector", collector_node)

workflow.add_edge("extract_title", "collector")
workflow.add_edge("extract_summary", "collector")
workflow.add_edge("extract_toc", "collector")
workflow.add_edge("extract_metadata", "collector")
workflow.add_edge("classify_doctype", "collector")
workflow.add_edge("extract_authors", "collector")

# The collector node transitions to the end
workflow.add_edge("collector", END)

# --- 3. Compile the Graph ---
app = workflow.compile()

# --- 4. Define a Sample Input Text ---
# A comprehensive sample document for testing
sample_text = """
# 城市大脑建设与运营管理标准

**发布日期**: 2023-11-01
**生效日期**: 2023-12-01
**作废日期**: 2033-12-01

**起草单位**: 未来城市研究中心
**作者**: 张三, 李四

**适用范围**: 本标准适用于中华人民共和国四川省成都市的城市大脑项目。

---

## **摘要**

本文档规定了城市大脑（City Brain）项目的建设、运营和管理的相关标准与要求，旨在确保项目的规范性、安全性和高效性。

---

## **目录**

1.  **引言**
    1.1. 背景
    1.2. 目的
2.  **核心技术要求**
    2.1. 数据融合平台
    2.2. AI算法引擎
    2.3. 安全体系
3.  **运营管理规范**
    3.1. 组织架构
    3.2. 应急预案
4.  **附录**
    4.1. 名词解释

---

## **1. 引言**

### **1.1. 背景**

随着信息技术的飞速发展，城市管理面临着前所未有的机遇与挑战。

### **1.2. 目的**

本标准的目的是为了统一和规范成都市城市大脑的建设与运营流程。

... (正文内容省略) ...
"""

# --- 5. Run the Graph ---
if __name__ == "__main__":
    print("🚀 Starting the document analysis process...")

    # The initial state for the graph
    initial_state = {"text": sample_text}

    # Invoke the graph with the initial state
    # The `stream` method provides real-time updates from each node
    final_state = app.invoke(initial_state)

    # Print the final, structured output
    print("\n\n✅ Document analysis complete!")
    print("--- Final Result ---")

    # Use json.dumps for pretty printing the dictionary
    # Ensure ensure_ascii=False to correctly display Chinese characters
    final_state_json = json.dumps(final_state, indent=2, ensure_ascii=False)
    print(final_state_json)

    # You can also access individual keys
    # print("\n--- Extracted Summary ---")
    # print(final_state.get('summary'))
