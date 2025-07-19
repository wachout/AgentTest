import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PreprocessorAgent:
    def __init__(self, max_chunk_size=20000, overlap=1000):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )

    def process(self, text: str) -> list[str]:
        """
        Processes the text, splitting it into chunks if it's too long.
        """
        if len(text) > self.max_chunk_size:
            return self.text_splitter.split_text(text)
        else:
            return [text]
