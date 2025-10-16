from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from .llm_client import get_llm

class TopicValidator:
    def __init__(self, provider="deepseek"):
        self.llm = get_llm(provider)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a topic validator. Your knowledge base is about technical support for software. Classify the user's query as either 'in-domain' or 'out-of-domain'. Respond in JSON format with 'classification' and 'confidence' keys. Query: {query}"
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def validate_topic(self, query: str):
        return self.chain.invoke({"query": query})