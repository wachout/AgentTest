import os
import json
import operator
from typing import List, TypedDict, Annotated, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Define the State for the graph
class AgentState(TypedDict):
    """
    Represents the state of our multi-agent analysis.
    """
    text: str # The original input text
    title: str # The extracted document title
    chunks: List[str] # Chunks of the text if it's too long
    summary: str # The extracted summary
    toc: Dict # The extracted table of contents as JSON
    metadata: Dict # Extracted metadata (dates, location)
    doc_type: str # The classified document type
    authors: List[str] # A list of extracted authors
    error: Annotated[List[str], operator.add] # A list to capture errors, combining them

# 2. Initialize the LLM
# We use ChatOpenAI because both DeepSeek and Alibaba provide OpenAI-compatible APIs.
# Make sure to have TONGYI_API_KEY in your .env file
api_key = os.getenv("TONGYI_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not api_key:
    raise ValueError("TONGYI_API_KEY not found in environment variables. Please set it in the .env file.")

# Using Alibaba Qwen model as requested
llm = ChatOpenAI(
    temperature=0.7,
    model="qwen-max", # Using a powerful model for complex tasks
    api_key=api_key,
    base_url=base_url,
)

# 3. Define Pydantic models for structured output
class TableOfContents(BaseModel):
    """Data model for a table of contents."""
    toc: Dict = Field(description="A hierarchical dictionary representing the table of contents. Keys are section titles, and values can be nested dictionaries for subsections. Return an empty dictionary if no ToC is found.")

class Metadata(BaseModel):
    """Data model for document metadata."""
    creation_date: str | None = Field(description="The creation date of the document.")
    update_date: str | None = Field(description="The last update date of the document.")
    expiration_date: str | None = Field(description="The expiration or廢止 date of the document.")
    effective_date: str | None = Field(description="The date the document becomes effective.")
    geographic_scope: str | None = Field(description="The geographic area the document applies to (e.g., '全国', '北京市', '四川省').")

class Authors(BaseModel):
    """Data model for a list of authors."""
    authors: List[str] = Field(description="A list of author names. Returns an empty list if no authors are found.")

# 4. Implement the Nodes for the Graph

def preprocess_text(state: AgentState) -> AgentState:
    """
    Checks the text length and chunks it if it exceeds 20,000 characters.
    """
    print("---(Node: Preprocessing Text)---")
    text = state.get("text", "")
    chunks = []
    if len(text) > 20000:
        print(f"Text is long ({len(text)} chars), chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=1000,
            add_start_index=True,
        )
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks.")
    else:
        print("Text is short enough, no chunking needed.")

    return {"chunks": chunks}


def extract_title(state: AgentState) -> AgentState:
    """
    Extracts the main title from the document.
    """
    print("---(Node: Extracting Title)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided to extract title."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in document analysis. Your task is to find and return the main title of the document."),
        ("user", "Please extract the main title from the following document. The title is usually at the very top and is the most prominent headline. Return only the title text.\n\nDocument:\n{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": text})
        title = response.content.strip()
        print(f"   Title extracted: {title}")
        return {"title": title}
    except Exception as e:
        print(f"   Error extracting title: {e}")
        return {"error": [f"Failed to extract title: {e}"]}


def extract_summary(state: AgentState) -> AgentState:
    """
    Extracts the summary or main idea from the document.
    For now, it processes the full text. A map-reduce approach on chunks would be an enhancement.
    """
    print("---(Node: Extracting Summary)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided to summarize."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in summarizing documents. Your task is to read the provided text and extract its main idea, summary, or abstract."),
        ("user", "Please provide a concise summary of the following document.\n\nDocument:\n{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": text})
        summary = response.content
        print(f"   Summary extracted.")
        return {"summary": summary}
    except Exception as e:
        print(f"   Error extracting summary: {e}")
        return {"error": [f"Failed to extract summary: {e}"]}


def extract_toc(state: AgentState) -> AgentState:
    """
    Extracts the table of contents as a JSON object.
    """
    print("---(Node: Extracting Table of Contents)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided for ToC extraction."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist in document structure analysis. Your goal is to find and parse the table of contents (目录) from the text."),
        ("user", "Analyze the following text and extract its table of contents. If a ToC is present, format it as a hierarchical JSON object. The JSON should represent the structure with keys for titles and nested objects for sub-sections. If no ToC is found, return an empty JSON object.\n\nText:\n{text}")
    ])

    structured_llm = llm.with_structured_output(TableOfContents)
    chain = prompt | structured_llm

    try:
        response = chain.invoke({"text": text})
        print(f"   ToC extracted.")
        return {"toc": response.toc}
    except Exception as e:
        print(f"   Error extracting ToC: {e}")
        return {"error": [f"Failed to extract ToC: {e}"]}

def extract_metadata(state: AgentState) -> AgentState:
    """
    Extracts metadata (dates, scope) from the document.
    """
    print("---(Node: Extracting Metadata)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided for metadata extraction."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specialized in extracting key metadata from documents."),
        ("user", "From the text below, extract the following metadata: the document's creation date (创作时间), update date (更新时间), expiration date (作废时间), effective date (生效时间), and its geographic scope (区域维度, e.g.,省,市级). Format the output as a JSON object. If a value is not found, set it to null.\n\nText:\n{text}")
    ])

    structured_llm = llm.with_structured_output(Metadata)
    chain = prompt | structured_llm

    try:
        response = chain.invoke({"text": text})
        print(f"   Metadata extracted.")
        return {"metadata": response.dict()}
    except Exception as e:
        print(f"   Error extracting metadata: {e}")
        return {"error": [f"Failed to extract metadata: {e}"]}


def classify_doctype(state: AgentState) -> AgentState:
    """
    Classifies the document type.
    """
    print("---(Node: Classifying Document Type)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided for classification."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document classification expert. Your job is to determine the type of a document based on its content."),
        ("user", "Please classify the type of the following document (e.g., '法律文书', '医学文档', '技术手册', '新闻文章', '研究报告'). Provide only the single best classification as a string.\n\nText:\n{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": text})
        doc_type = response.content.strip()
        print(f"   Document type classified as: {doc_type}")
        return {"doc_type": doc_type}
    except Exception as e:
        print(f"   Error classifying document: {e}")
        return {"error": [f"Failed to classify document: {e}"]}


def extract_authors(state: AgentState) -> AgentState:
    """
    Extracts the authors of the document.
    """
    print("---(Node: Extracting Authors)---")
    text = state.get("text", "")
    if not text:
        return {"error": ["No text provided for author extraction."]}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that extracts author names from texts."),
        ("user", "Extract the author(s) of this document. If there are multiple authors, list them all. Format the output as a JSON list of strings. If no authors are found, return an empty list.\n\nText:\n{text}")
    ])

    structured_llm = llm.with_structured_output(Authors)
    chain = prompt | structured_llm

    try:
        response = chain.invoke({"text": text})
        print(f"   Authors extracted.")
        return {"authors": response.authors}
    except Exception as e:
        print(f"   Error extracting authors: {e}")
        return {"error": [f"Failed to extract authors: {e}"]}
