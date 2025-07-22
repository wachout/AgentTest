from src.tools.chapter_identifier import identify_chapters
from src.tools.thematic_identifier import identify_thematic_shifts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import datetime

class DocumentMetadata(BaseModel):
    """Represents the metadata of the document."""
    document_type: str = Field(description="The type of the document (e.g., article, report, email).")
    business_area: str = Field(description="The business area the document belongs to (e.g., finance, marketing, legal).")
    region: str = Field(description="The geographical region the document is relevant to (e.g., global, US, EMEA).")

class KnowledgeArchitectAgent:
    def __init__(self, llm):
        self.llm = llm

    def analyze_text(self, text: str) -> dict:
        """
        Analyzes the text to determine the three-tiered knowledge structure.

        Args:
            text: The text to analyze.

        Returns:
            A dictionary representing the knowledge structure.
        """
        metadata = self._extract_metadata(text)
        chapters = identify_chapters(self.llm, text)
        thematic_sections = identify_thematic_shifts(self.llm, text)

        return {
            "metadata": {
                "document_type": metadata.document_type,
                "business_area": metadata.business_area,
                "timestamps": {
                    "created": datetime.datetime.now().isoformat(),
                    "updated": datetime.datetime.now().isoformat(),
                    "obsoleted": None,
                },
                "region": metadata.region,
            },
            "chapters": chapters,
            "thematic_sections": thematic_sections,
        }

    def _extract_metadata(self, text: str) -> DocumentMetadata:
        """
        Extracts metadata from the text using an LLM.

        Args:
            text: The text to analyze.

        Returns:
            A DocumentMetadata object.
        """
        parser = JsonOutputParser(pydantic_object=DocumentMetadata)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in document analysis. Your task is to extract the metadata from the given text. Respond with a JSON object containing the document type, business area, and region."),
            ("user", "{text}")
        ])

        chain = prompt | self.llm | parser

        try:
            return chain.invoke({"text": text})
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return DocumentMetadata(document_type="unknown", business_area="unknown", region="unknown")
