from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Chapter(BaseModel):
    """Represents a single chapter in the document."""
    start: int = Field(description="The character offset where the chapter starts.")
    end: int = Field(description="The character offset where the chapter ends.")
    title: str = Field(description="The title of the chapter.")

class Chapters(BaseModel):
    """A list of chapters in the document."""
    chapters: list[Chapter]

def identify_chapters(llm, text: str) -> list[int]:
    """
    Identifies the chapter breaks in a given text.

    Args:
        llm: The language model to use.
        text: The text to analyze.

    Returns:
        A list of character offsets where chapter breaks occur.
    """
    parser = JsonOutputParser(pydantic_object=Chapters)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in document analysis. Your task is to identify the chapter breaks in the given text. Respond with a JSON object containing a list of chapters, where each chapter has a start offset, an end offset, and a title."),
        ("user", "{text}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"text": text})
        return [chapter['start'] for chapter in result['chapters']]
    except Exception as e:
        print(f"Error identifying chapters: {e}")
        return []
