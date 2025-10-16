from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm

class ResponseGenerator:
    def __init__(self, provider="deepseek"):
        self.llm = get_llm(provider)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Generate a concise and informative answer to the user's query based on the provided context. If the context is not relevant, say so. Query: {query}\n\nContext:\n{context}"
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, query: str, context: list):
        context_str = "\n".join([doc["content"] for doc in context])
        return self.chain.invoke({"query": query, "context": context_str})