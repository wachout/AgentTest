import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from typing import List

load_dotenv()

class ThemeSplitPoint(BaseModel):
    position: int = Field(description="切割点的字符位置")
    theme: str = Field(description="主旨语义")

class ThemeSplitResult(BaseModel):
    split_points: List[ThemeSplitPoint] = Field(description="主旨语义切割点列表")

class ThemeAgent:
    def __init__(self, model_name="deepseek-reasoner"):
        if model_name == "deepseek-reasoner":
            self.llm = ChatOpenAI(
                temperature=0.6,
                model="deepseek-reasoner",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )
        elif model_name == "qwen3-32b":
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="qwen3-32b",
                api_key=os.getenv("QWEN_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            raise ValueError("Unsupported model name")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个专业的文本分析师。你的任务是根据提供的文本，分析出文本按主旨语义的分割点，并输出该切割点的位置和主旨语义。",
                ),
                ("user", "文本内容：\n\n{text}"),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(ThemeSplitResult)

    async def analyze(self, text: str) -> ThemeSplitResult:
        """
        Analyzes the text to find theme split points.
        """
        return await self.chain.ainvoke({"text": text})
