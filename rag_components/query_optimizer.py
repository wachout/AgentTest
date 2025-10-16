from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm

class QueryOptimizer:
    def __init__(self, provider="deepseek"):
        self.llm = get_llm(provider)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a query optimizer. Your task is to refine the user's query to improve search results. Generate a new, more specific query based on the original. Original Query: {query}"
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def optimize_query(self, query: str):
        return self.chain.invoke({"query": query})