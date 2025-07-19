import asyncio
import json
from src.graph import TextAnalysisGraph

async def main(text: str, model_name: str = "deepseek-reasoner"):
    """
    Asynchronously analyzes the text and prints the result.
    """
    graph_builder = TextAnalysisGraph(model_name)
    workflow = graph_builder.build_graph()

    initial_state = {"text": text, "error": None}

    final_state = await workflow.ainvoke(initial_state)

    print(json.dumps(final_state.get("aggregated_result"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Example usage with a short text
    short_text = """
    第一章：总则
    第一条 为了规范公司的财务行为，加强财务管理，根据国家有关法律、法规及公司章程的规定，结合公司的实际情况，特制定本制度。
    第二章：财务机构与人员
    第二条 公司设立财务部，负责公司的财务管理工作。财务部设经理一名，会计若干名。
    """
    asyncio.run(main(short_text))

    # To run with a long text, you would replace the short_text with your long text.
    # For example:
    # with open("long_text.txt", "r", encoding="utf-8") as f:
    #     long_text = f.read()
    # asyncio.run(main(long_text))
