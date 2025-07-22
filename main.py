import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi

from src.graph.workflow import create_workflow

def main():
    """
    The main entry point for the application.
    """
    load_dotenv()

    # Initialize the LLMs
    deepseek_llm = ChatOpenAI(
        temperature=0.6,
        model="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    tongyi_llm = ChatTongyi(
        temperature=0.7,
        model="qwen3-32b",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    tongyi_llm = tongyi_llm.bind(enable_thinking=False)

    # Create the workflow
    app = create_workflow(deepseek_llm)

    # Example usage
    text = "This is a sample text. It has multiple chapters and thematic sections."
    inputs = {"text": text}
    result = app.invoke(inputs)
    print(result)

if __name__ == "__main__":
    main()
