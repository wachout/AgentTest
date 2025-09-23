from pymilvus import connections, Collection
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBEDDING_DIM

class MilvusClient:
    def __init__(self):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            print("Successfully connected to Milvus.")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

        if not self._collection_exists(COLLECTION_NAME):
            raise Exception(f"Collection '{COLLECTION_NAME}' does not exist in Milvus.")

        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()

    def _collection_exists(self, collection_name):
        # A helper to check if a collection exists
        from pymilvus import utility
        return utility.has_collection(collection_name)

    def search(self, vector, top_k=5):
        """
        Searches for similar vectors in both title and content embeddings.
        """
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        # Search in title embedding
        title_results = self.collection.search(
            data=[vector],
            anns_field="title_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content"]
        )

        # Search in content embedding
        content_results = self.collection.search(
            data=[vector],
            anns_field="content_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content"]
        )

        # Combine and deduplicate results
        all_results = []
        seen_ids = set()

        # Process results
        for hits in [title_results[0], content_results[0]]:
            for hit in hits:
                if hit.id not in seen_ids:
                    all_results.append({
                        "id": hit.id,
                        "score": hit.distance,
                        "title": hit.entity.get('title'),
                        "content": hit.entity.get('content')
                    })
                    seen_ids.add(hit.id)

        # Sort by score (distance)
        all_results.sort(key=lambda x: x["score"])

        return all_results

# Singleton instance
milvus_client = MilvusClient()
