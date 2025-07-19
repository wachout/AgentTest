import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class ArticleMetadata(BaseModel):
    article_type: str = Field(description="文章类型，例如：新闻、报告、政策文件等")
    business_type: str = Field(description="业务类型，例如：金融、医疗、科技等")
    creation_date: str = Field(description="文章的创作时间")
    update_date: str = Field(description="文章的更新时间")
    expiration_date: str = Field(description="文章的作废时间")
    region: str = Field(description="区域维度，例如：省级、市级")

class MetadataAgent:
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
                    "你是一个专业的文本分析师。你的任务是根据提供的文本，提取以下元数据：文章类型、业务类型、创作/更新/作废时间、区域维度。",
                ),
                ("user", "文本内容：\n\n{text}"),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(ArticleMetadata)

    async def analyze(self, text: str) -> ArticleMetadata:
        """
        Analyzes the text to extract metadata.
        """
        return await self.chain.ainvoke({"text": text})
