from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from .llm_client import get_llm
from .state import AgentState

def validate_topic(state: AgentState):
    """
    Validates if the query is in-domain.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a topic validator. Your knowledge base is about technical support for software. Classify the user's query as either 'in_domain' or 'out_of_domain'. Respond in JSON format with a 'classification' key. Query: {query}"
    )
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    classification = chain.invoke({"query": state["enhanced_query"]})
    return {"topic_classification": classification}