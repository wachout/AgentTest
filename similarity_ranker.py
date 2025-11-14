from sentence_transformers import SentenceTransformer, util

def rank_texts_by_similarity(query, texts):
    """
    Calculates the similarity between a query and a list of texts,
    and returns the texts ranked by similarity.

    Args:
        query (str): The input query string.
        texts (list): A list of text strings to compare against the query.

    Returns:
        list: A list of tuples, where each tuple contains a text and its
              similarity score, sorted in descending order of similarity.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the query and texts
    query_embedding = model.encode(query, convert_to_tensor=True)
    text_embeddings = model.encode(texts, convert_to_tensor=True)

    # Calculate cosine similarity between the query and each text
    cosine_scores = util.pytorch_cos_sim(query_embedding, text_embeddings)

    # Create a list of (text, score) tuples
    ranked_texts = []
    for i in range(len(texts)):
        ranked_texts.append((texts[i], cosine_scores[0][i].item()))

    # Sort the list by score in descending order
    ranked_texts.sort(key=lambda x: x[1], reverse=True)

    return ranked_texts

if __name__ == '__main__':
    # Example usage
    sample_query = "A cute, fluffy dog."
    sample_texts = [
        "A small, adorable cat.",
        "A large, playful dog.",
        "A book about programming.",
        "A fluffy, brown puppy."
    ]

    ranked_results = rank_texts_by_similarity(sample_query, sample_texts)

    print(f"Query: {sample_query}\n")
    print("Ranked Texts (most to least similar):")
    for text, score in ranked_results:
        print(f"- {text} (Score: {score:.4f})")