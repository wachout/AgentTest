from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 20000, chunk_overlap: int = 1000) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The amount of overlap between chunks.

    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)
