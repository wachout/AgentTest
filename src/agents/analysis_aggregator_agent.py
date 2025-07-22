from src.agents.text_segmentation_agent import TextSegmentationAgent
from src.agents.knowledge_architect_agent import KnowledgeArchitectAgent

class AnalysisAggregatorAgent:
    def __init__(self, llm):
        self.text_segmentation_agent = TextSegmentationAgent()
        self.knowledge_architect_agent = KnowledgeArchitectAgent(llm)

    def analyze_document(self, text: str) -> dict:
        """
        Analyzes a document by splitting it into chunks, analyzing each chunk,
        and aggregating the results.

        Args:
            text: The document to analyze.

        Returns:
            A dictionary representing the aggregated analysis.
        """
        chunks = self.text_segmentation_agent.segment_text(text)

        aggregated_results = {
            "metadata": None,
            "chapters": [],
            "thematic_sections": [],
        }

        for i, chunk in enumerate(chunks):
            analysis = self.knowledge_architect_agent.analyze_text(chunk)
            if i == 0:
                aggregated_results["metadata"] = analysis["metadata"]

            # Adjust the chapter and thematic section offsets based on the chunk's position
            chunk_offset = i * (20000 - 1000)
            aggregated_results["chapters"].extend([offset + chunk_offset for offset in analysis["chapters"]])
            aggregated_results["thematic_sections"].extend([offset + chunk_offset for offset in analysis["thematic_sections"]])

        return aggregated_results
