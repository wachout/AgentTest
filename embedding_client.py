from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class EmbeddingClient:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingClient, cls).__new__(cls)
            # Load the model only once
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance

    def get_embedding(self, text):
        """
        Generates an embedding for the given text.
        """
        if self._model is None:
            raise Exception("Model is not loaded.")

        # The model produces a list of embeddings, one for each input text.
        # Since we are passing a single text, we take the first element.
        embedding = self._model.encode(text)
        return embedding.tolist()

# Singleton instance
embedding_client = EmbeddingClient()
