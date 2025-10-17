import os
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "tech_support_kb"
DIMENSION = 384 # Based on the 'all-MiniLM-L6-v2' model

def main():
    # Connect to Milvus
    connections.connect(alias="default", uri=MILVUS_URI)
    print("Connected to Milvus.")

    # Remove old collection if it exists
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped collection: {COLLECTION_NAME}")

    # Create collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Technical support knowledge base")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection: {COLLECTION_NAME}")

    # Create index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    print("Created index.")

    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample documents with titles
    documents = [
        {"title": "Password Reset", "content": "To reset your password, go to the settings page and click on 'Reset Password'."},
        {"title": "Profile Update", "content": "You can update your profile from the 'Profile' section in your account."},
        {"title": "Billing Support", "content": "For any billing issues, please contact our support team at support@example.com."},
        {"title": "OS Compatibility", "content": "Our software is compatible with Windows, macOS, and Linux."},
        {"title": "Refund Policy", "content": "To get a refund, you must request it within 30 days of purchase."}
    ]

    # Insert data
    titles = [doc["title"] for doc in documents]
    contents = [doc["content"] for doc in documents]
    embeddings = model.encode(contents)
    data = [embeddings, titles, contents]
    collection.insert(data)
    collection.flush()
    print(f"Inserted {len(documents)} documents into the collection.")

    # Load collection for searching
    collection.load()
    print("Collection loaded.")

if __name__ == "__main__":
    main()