import os
import json
import jieba
from flask import Flask, request, jsonify, render_template
from neo4j import GraphDatabase
from pymilvus import connections, Collection
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# --- Neo4j Connection ---
class Neo4jQuerier:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def find_subchapters_by_keywords(self, keywords):
        with self.driver.session() as session:
            result = session.execute_read(self._find_and_return_subchapters, keywords)
            return result

    @staticmethod
    def _find_and_return_subchapters(tx, keywords):
        query = (
            "UNWIND $keywords AS keyword "
            "MATCH (k:Keyword) WHERE k.name CONTAINS keyword "
            "MATCH (k)<-[:HAS_KEYWORD]-(sc:SubChapter) "
            "RETURN DISTINCT sc.sub_title AS sub_title, sc.sub_content AS sub_content"
        )
        result = tx.run(query, keywords=keywords)
        return [{"sub_title": record["sub_title"], "sub_content": record["sub_content"]} for record in result]

# --- Milvus Connection & Vector Search ---
class VectorSearcher:
    def __init__(self, host, port, collection_name, model_name='bert-base-chinese'):
        connections.connect("default", host=host, port=port)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def search(self, query_vector, top_k=5):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query_vector,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["sub_title"]
        )
        return [hit.entity.get('sub_title') for hit in results[0]]

# --- Global Variables & Initialization ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
neo4j_querier = Neo4jQuerier(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
COLLECTION_NAME = "law_embeddings"
# We will initialize vector_searcher later inside a request context or a dedicated init function
# to avoid issues with Flask's reloader and multiple initializations.
vector_searcher = None

def init_vector_searcher():
    global vector_searcher
    if vector_searcher is None:
        try:
            vector_searcher = VectorSearcher(MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME)
        except Exception as e:
            print(f"Error initializing VectorSearcher: {e}")
            vector_searcher = None


@app.route("/")
def index():
    return render_template("index.html")


@app.before_request
def before_first_request():
    init_vector_searcher()


@app.route("/search", methods=["POST"])
def search():
    if not request.json or 'query' not in request.json:
        return jsonify({"error": "Query not provided"}), 400

    query = request.json['query']

    if vector_searcher is None:
        return jsonify({"error": "Vector searcher not initialized. Check Milvus connection."}), 503

    # 1. Keyword Extraction
    keywords = list(jieba.cut_for_search(query))

    # 2. Graph-based Candidate Retrieval (Neo4j)
    try:
        candidate_docs = neo4j_querier.find_subchapters_by_keywords(keywords)
        if not candidate_docs:
            # Fallback or alternative strategy if no keywords match
            # For now, we'll return empty
             return jsonify([])
    except Exception as e:
        return jsonify({"error": f"Neo4j query failed: {e}"}), 500


    # 3. Vector-based Re-ranking (Milvus)
    query_embedding = vector_searcher.get_embedding(query)

    candidate_embeddings = {
        doc['sub_title']: vector_searcher.get_embedding(doc['sub_content'])
        for doc in candidate_docs
    }

    # Calculate cosine similarity
    scores = {}
    for title, emb in candidate_embeddings.items():
        # Cosine similarity calculation
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.from_numpy(query_embedding),
            torch.from_numpy(emb)
        ).item()
        scores[title] = cos_sim

    # Sort candidates by score
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # 4. Format and Return Results
    results = [{"sub_title": title, "score": score} for title, score in sorted_candidates]

    return jsonify(results)

if __name__ == "__main__":
    # In a real production environment, use a Gunicorn or Waitress server.
    app.run(host="0.0.0.0", port=5001, debug=True)
