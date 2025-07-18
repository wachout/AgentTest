from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class ThematicSection(BaseModel):
    """Represents a single thematic section in the document."""
    start: int = Field(description="The character offset where the thematic section starts.")
    end: int = Field(description="The character offset where the thematic section ends.")
    theme: str = Field(description="The theme of the section.")

class ThematicSections(BaseModel):
    """A list of thematic sections in the document."""
    sections: list[ThematicSection]

def identify_thematic_shifts(llm, text: str) -> list[int]:
    """
    Identifies the thematic shifts in a given text.

    Args:
        llm: The language model to use.
        text: The text to analyze.

    Returns:
        A list of character offsets where thematic shifts occur.
    """
    parser = JsonOutputParser(pydantic_object=ThematicSections)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in document analysis. Your task is to identify the thematic shifts in the given text. Respond with a JSON object containing a list of thematic sections, where each section has a start offset, an end offset, and a theme."),
        ("user", "{text}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"text": text})
        return [section['start'] for section in result['sections']]
    except Exception as e:
        print(f"Error identifying thematic shifts: {e}")
        return []
