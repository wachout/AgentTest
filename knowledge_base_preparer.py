import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 加载环境变量 (例如，如果使用OpenAI API)
load_dotenv()

def create_vector_store():
    """
    创建并保存文本知识库的FAISS向量索引。
    """
    # 定义知识库文件和FAISS索引的路径
    knowledge_base_file = 'data/knowledge_base.txt'
    faiss_index_path = 'faiss_index'

    # 1. 加载文本文档
    print("正在加载文本文档...")
    loader = TextLoader(knowledge_base_file, encoding='utf-8')
    documents = loader.load()

    # 2. 切分文档
    # 我们将文档分割成更小的块，以便进行有效的嵌入和检索
    print("正在切分文档...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 3. 初始化嵌入模型
    # 这里我们使用OllamaEmbeddings。请确保你已经安装并运行了Ollama，
    # 并且已经拉取了所需的模型 (例如: ollama run nomic-embed-text)
    # 如果你想使用OpenAI，可以替换成下面的代码：
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    print("正在初始化嵌入模型...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. 创建FAISS向量存储
    # 这会为所有文档块创建嵌入向量，并存储在一个高效的索引结构中
    print("正在创建FAISS向量存储...")
    try:
        db = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        print(f"创建FAISS索引时出错: {e}")
        print("请确保你的嵌入模型正在运行并且配置正确。")
        print("如果你在使用Ollama，请确认Ollama服务已启动，并且模型 'nomic-embed-text' 已被拉取。")
        return

    # 5. 保存FAISS索引到本地
    # 这样我们就不需要在每次运行时都重新创建它
    print(f"正在将FAISS索引保存到 '{faiss_index_path}'...")
    db.save_local(faiss_index_path)

    print("\n知识库向量索引创建并保存成功！")
    print(f"索引保存在: {os.path.abspath(faiss_index_path)}")


if __name__ == '__main__':
    # 检查FAISS索引是否已存在
    if os.path.exists('faiss_index'):
        print("FAISS索引目录 'faiss_index' 已存在。")
        user_input = input("是否要重新创建索引？(y/n): ").lower()
        if user_input == 'y':
            import shutil
            shutil.rmtree('faiss_index')
            print("已删除旧的索引。")
            create_vector_store()
        else:
            print("操作取消。")
    else:
        create_vector_store()