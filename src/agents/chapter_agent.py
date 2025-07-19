import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from typing import List

load_dotenv()

class ChapterSplitPoint(BaseModel):
    position: int = Field(description="切割点的字符位置")
    title: str = Field(description="章节标题")

class ChapterSplitResult(BaseModel):
    split_points: List[ChapterSplitPoint] = Field(description="章节切割点列表")

class ChapterAgent:
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
                    "你是一个专业的文本分析师。你的任务是根据提供的文本，分析出文本按章节的分割点，并输出该切割点的位置和章节标题。",
                ),
                ("user", "文本内容：\n\n{text}"),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(ChapterSplitResult)

    async def analyze(self, text: str) -> ChapterSplitResult:
        """
        Analyzes the text to find chapter split points.
        """
        return await self.chain.ainvoke({"text": text})
