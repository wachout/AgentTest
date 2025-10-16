from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from .llm_client import get_llm

class RelevanceAssessor:
    def __init__(self, provider="deepseek"):
        self.llm = get_llm(provider)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a relevance assessor. Your task is to determine if the retrieved documents are relevant to the user's query. Respond in JSON format with 'is_relevant' (boolean) and 'reasoning' (string). Query: {query}\n\nDocuments:\n{documents}"
        )
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def assess_relevance(self, query: str, documents: list):
        doc_content = "\n".join([doc["content"] for doc in documents])
        return self.chain.invoke({"query": query, "documents": doc_content})