import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv()

def get_deepseek_llm():
    """Returns the DeepSeek LLM client."""
    return ChatOpenAI(
        temperature=0.6,
        model="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

def get_tongyi_llm():
    """Returns the Tongyi LLM client."""
    return ChatTongyi(
        temperature=0.7,
        model="qwen2-72b-instruct",
        api_key=os.getenv("TONGYI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def get_llm(provider="deepseek"):
    """
    Returns the specified LLM client.

    :param provider: "deepseek" or "tongyi"
    :return: The LLM client.
    """
    if provider == "deepseek":
        llm = get_deepseek_llm()
    elif provider == "tongyi":
        llm = get_tongyi_llm()
    else:
        raise ValueError("Invalid LLM provider specified.")

    return llm