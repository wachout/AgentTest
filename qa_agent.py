from embedding_client import embedding_client
from milvus_client import milvus_client

def ask(question: str):
    """
    Takes a question, searches for relevant documents in Milvus, and returns them.
    """
    print(f"Received question: {question}")

    # 1. Generate embedding for the question
    question_embedding = embedding_client.get_embedding(question)
    print("Successfully generated question embedding.")

    # 2. Search for relevant documents in Milvus
    try:
        search_results = milvus_client.search(question_embedding)
        print("Successfully searched Milvus.")
        return search_results
    except Exception as e:
        print(f"An error occurred during Milvus search: {e}")
        return None

def format_results(results):
    """
    Formats the search results for display.
    """
    if not results:
        return "No relevant documents found."

    formatted_string = "Found relevant documents:\n"
    for result in results:
        formatted_string += f"\n--- Result ---\n"
        formatted_string += f"Title: {result['title']}\n"
        formatted_string += f"Content: {result['content']}\n"
        formatted_string += f"Score: {result['score']:.4f}\n"

    return formatted_string

if __name__ == "__main__":
    # This is for testing purposes, as outlined in the next step.
    sample_question = "What is the capital of France?" # Replace with a relevant question for your data

    results = ask(sample_question)

    if results:
        print(format_results(results))
