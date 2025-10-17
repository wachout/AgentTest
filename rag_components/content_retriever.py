import os
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from .state import AgentState

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "tech_support_kb"
DIMENSION = 384

connections.connect(alias="default", uri=MILVUS_URI)
model = SentenceTransformer('all-MiniLM-L6-v2')
collection = Collection(name=COLLECTION_NAME)
collection.load()

def retrieve_content(state: AgentState):
    """
    Retrieves content from Milvus using similarity search.
    """
    query_embedding = model.encode([state["enhanced_query"]])[0]

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["title", "content"]
    )

    retrieved_docs = []
    for hit in results[0]:
        retrieved_docs.append({
            "title": hit.entity.get("title"),
            "content": hit.entity.get("content"),
            "score": hit.distance
        })

    return {"retrieved_docs": retrieved_docs}