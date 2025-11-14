import faiss
import numpy as np
import pickle

FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"

class VectorStore:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved_documents = [{"text": self.chunks[i].text, "distance": float(d)} for d, i in zip(distances[0], indices[0])]
        return retrieved_documents