from src.tools.text_splitter import split_text

class TextSegmentationAgent:
    def __init__(self):
        pass

    def segment_text(self, text: str) -> list[str]:
        """
        Segments a long text into smaller, overlapping chunks.

        Args:
            text: The text to segment.

        Returns:
            A list of text chunks.
        """
        return split_text(text)
