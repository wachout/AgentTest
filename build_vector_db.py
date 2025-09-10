import json
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

# --- Milvus 连接信息 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# --- Collection 配置 ---
COLLECTION_NAME = "legal_documents"
ID_FIELD_NAME = "doc_id"
VECTOR_FIELD_NAME = "vector"

# --- 模型配置 ---
MODEL_NAME = 'bge-small-zh-v1.5' # 一个流行的中文 embedding 模型

def connect_to_milvus():
    """连接到 Milvus 服务"""
    print("Connecting to Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus.")

def create_milvus_collection(dim):
    """创建 Milvus Collection"""
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it.")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name=ID_FIELD_NAME, dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Collection for legal document search")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    print(f"Collection '{COLLECTION_NAME}' created successfully.")
    return collection

def create_index(collection):
    """为向量字段创建索引"""
    print("Creating index for vector field...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name=VECTOR_FIELD_NAME, index_params=index_params)
    print("Index created successfully.")

def vectorize_and_insert(collection, data, model):
    """将数据向量化并插入 Milvus"""
    print("Vectorizing and inserting data...")

    sub_chapters = data.get('sub_chapters', [])
    if not sub_chapters:
        print("No sub_chapters found in the data.")
        return

    # 提取所有 sub_content 用于批量编码
    contents = [item['sub_content'] for item in sub_chapters]
    ids = [item['sub_title'] for item in sub_chapters]

    # 批量生成向量
    vectors = model.encode(contents, show_progress_bar=True)

    # 准备插入数据
    entities = [
        ids,
        vectors
    ]

    # 插入数据
    collection.insert(entities)
    collection.flush() # 确保数据被写入磁盘

    print(f"Successfully inserted {len(ids)} vectors into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    # 1. 加载预训练模型
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Vector dimension: {embedding_dim}")

    # 2. 连接到 Milvus
    # connect_to_milvus() # 在实际环境中取消注释

    # 3. 创建 Collection
    # collection = create_milvus_collection(embedding_dim) # 在实际环境中取消注释

    # 4. 加载 JSON 数据
    print("Loading data from data.json...")
    with open('data.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 5. 向量化并插入数据
    # vectorize_and_insert(collection, json_data, model) # 在实际环境中取消注释

    # 6. 为集合创建索引并加载
    # create_index(collection) # 在实际环境中取消注释
    # collection.load() # 在实际环境中取消注释

    print("\n--- Script execution finished ---")
    print("NOTE: This script is in 'dry-run' mode.")
    print("To actually connect to Milvus and import data, you need to:")
    print("1. Start your Milvus instance.")
    print("2. Uncomment the function calls in the `if __name__ == '__main__':` block.")

    # connections.disconnect("default") # 在实际环境中取消注释
