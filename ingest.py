import os
from dotenv import load_dotenv
from lightrag.core.generator import Generator
from lightrag.core.embedder import Embedder
from lightrag.components.data_process.text_splitter import TextSplitter
# from lightrag.core.knowledge_graph import Neo4jGraph
import faiss
import numpy as np
import pickle
from lightrag.core.component import Component
from lightrag.core.types import Document

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# LLM for generation (entity extraction)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

# Embedding model
EMBEDDING_MODEL_NAME = "thenlper/gte-base"

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Data path
DATA_PATH = "data/sample.txt"
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"

# --- Entity Extraction Prompt ---
ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
From the text below, extract the following types of entities and their relationships related to "The Wandering Earth":
- Person (e.g., Dr. Liu)
- Organization (e.g., United Earth Government, UEG)
- Project (e.g., Earth Engine project)
- Concept (e.g., Great Catastrophe, Reining Age)

And the following relationships:
- WORKS_FOR (Person, Organization)
- MEMBER_OF (Person, Project)
- ASSOCIATED_WITH (Person, Concept)
- LEADS (Person, Project)

Respond with a list of JSON objects, where each object represents a relationship with a source entity, a target entity, and the relationship type.
Example:
[
  {
    "source": {"name": "Dr. Liu", "type": "Person"},
    "target": {"name": "Earth Engine project", "type": "Project"},
    "relationship": "LEADS"
  }
]

Text:
{text}
"""

class EntityExtractor(Component):
    def __init__(self):
        super().__init__()
        self.generator = Generator(
            model_name=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            prompt_template=ENTITY_EXTRACTION_PROMPT_TEMPLATE,
            response_format={"type": "json_object"}
        )

    def call(self, text: str) -> list:
        response = self.generator.call(text=text)
        return response.data

def ingest_data():
    """
    Full data ingestion pipeline.
    """
    print("Starting data ingestion...")

    # 1. Load data
    print("Loading data...")
    with open(DATA_PATH, "r") as f:
        text = f.read()
    documents = [Document(text=text)]

    # 2. Split documents into chunks
    print("Splitting documents...")
    splitter = TextSplitter(split_by="sentence", chunk_size=256, chunk_overlap=50)
    chunks = splitter.call(documents)
    print(f"Split into {len(chunks)} chunks.")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to {CHUNKS_PATH}")


    # 3. Extract entities and relationships from each chunk
    print("Extracting entities and relationships...")
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found. Skipping entity extraction.")
        all_triples = []
    else:
        entity_extractor = EntityExtractor()
        all_triples = []
        for chunk in chunks:
            triples = entity_extractor.call(chunk.text)
            if triples:
                all_triples.extend(triples)
    print(f"Extracted {len(all_triples)} triples.")

    # 4. Initialize Neo4j Graph and add triples
    print("Initializing Neo4j graph...")
    print("Neo4j credentials not found. Skipping graph ingestion.")


    # 5. Create vector index
    print("Creating vector index...")
    from lightrag.components.model_client import TransformersClient

    client = TransformersClient(model_name=EMBEDDING_MODEL_NAME)
    embedder = Embedder(model_client=client)
    response = embedder.call([chunk.text for chunk in chunks], model_kwargs={"model": EMBEDDING_MODEL_NAME})
    embeddings = np.array([item.embedding for item in response.data])

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")


    print("Data ingestion finished.")

if __name__ == "__main__":
    ingest_data()