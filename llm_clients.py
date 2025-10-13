import os
from langchain_openai import ChatOpenAI
from knowledge_graph_maker.types import LLMClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class DeepSeekClient(LLMClient):
    """
    用于与 DeepSeek 模型交互的客户端。
    """
    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.6, top_p: float = 0.5):
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self.llm = ChatOpenAI(
            temperature=self._temperature,
            model=self._model,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            top_p=self._top_p
        )

    def generate(self, user_message: str, system_message: str) -> str:
        """
        使用 DeepSeek 模型生成响应。
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        response = self.llm.invoke(messages)
        return response.content


class TongyiClient(LLMClient):
    """
    用于与通义千问模型交互的客户端。
    """
    def __init__(self, model: str = "qwen-plus", temperature: float = 0.7, top_p: float = 0.5):
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self.llm = ChatOpenAI(
            temperature=self._temperature,
            model=self._model,
            api_key=os.getenv("TONGYI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            top_p=self._top_p
        )
        self.llm = self.llm.bind(enable_thinking=False)

    def generate(self, user_message: str, system_message: str) -> str:
        """
        使用通义千问模型生成响应。
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        response = self.llm.invoke(messages)
        return response.content