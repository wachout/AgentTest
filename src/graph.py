from typing import List, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END
from src.agents.preprocessor_agent import PreprocessorAgent
from src.agents.metadata_agent import MetadataAgent
from src.agents.chapter_agent import ChapterAgent
from src.agents.theme_agent import ThemeAgent
from src.agents.aggregator_agent import AggregatorAgent

class GraphState(TypedDict):
    text: str
    chunks: List[str]
    results: List[Dict[str, Any]]
    aggregated_result: Dict[str, Any]
    error: str


class TextAnalysisGraph:
    def __init__(self, model_name="deepseek-reasoner"):
        self.model_name = model_name

    def _preprocess_text(self, state: GraphState):
        preprocessor = PreprocessorAgent()
        chunks = preprocessor.process(state["text"])
        return {"chunks": chunks, "results": []}

    async def _analyze_chunk(self, state: GraphState):
        metadata_agent = MetadataAgent(self.model_name)
        chapter_agent = ChapterAgent(self.model_name)
        theme_agent = ThemeAgent(self.model_name)

        results = []
        for chunk in state["chunks"]:
            metadata = await metadata_agent.analyze(chunk)
            chapters = await chapter_agent.analyze(chunk)
            themes = await theme_agent.analyze(chunk)
            results.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                    "chapters": chapters,
                    "themes": themes,
                }
            )
        return {"results": results}

    def _aggregate_results(self, state: GraphState):
        if len(state["chunks"]) > 1:
            aggregator = AggregatorAgent()
            aggregated_result = aggregator.aggregate(state["results"])
            return {"aggregated_result": aggregated_result}
        else:
            return {"aggregated_result": state["results"][0]}

    def _should_aggregate(self, state: GraphState):
        return len(state["chunks"]) > 1

    def build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("preprocess", self._preprocess_text)
        workflow.add_node("analyze", self._analyze_chunk)
        workflow.add_node("aggregate", self._aggregate_results)

        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            lambda state: "aggregate" if self._should_aggregate(state) else END,
            {"aggregate": "aggregate", END: END},
        )
        workflow.add_edge("aggregate", END)

        return workflow.compile()
