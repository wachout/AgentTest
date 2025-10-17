from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm
from .state import AgentState

def enhance_query(state: AgentState):
    """
    Enhances the user's query with chat history.
    """
    if not state["chat_history"]:
        enhanced_query = state["question"]
    else:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query enhancer. Your task is to rephrase the user's question to be a standalone question, integrating the context from the chat history."),
            ("user", "Chat History:\n{chat_history}\n\nQuestion:\n{question}")
        ])
        chain = prompt | llm | StrOutputParser()

        chat_history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in state["chat_history"]])

        enhanced_query = chain.invoke({
            "chat_history": chat_history_str,
            "question": state["question"]
        })

    return {"enhanced_query": enhanced_query}