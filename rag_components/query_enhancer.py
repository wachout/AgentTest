from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm

class QueryEnhancer:
    def __init__(self, provider="deepseek"):
        self.llm = get_llm(provider)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query enhancer. Your task is to rephrase the user's question to be a standalone question, integrating the context from the chat history. If the question is the first one, just return it as is."),
            ("user", "Chat History:\n{chat_history}\n\nQuestion:\n{question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def enhance_query(self, state):
        if not state.chat_history:
            return state.get_last_user_message()

        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.chat_history])

        return self.chain.invoke({
            "chat_history": chat_history_str,
            "question": state.get_last_user_message()
        })