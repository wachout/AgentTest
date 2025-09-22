import os
from typing import List, Optional
from dataclasses import dataclass, field, asdict

# Corrected imports based on library source code
from lightrag.core.generator import Generator
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client.openai_client import OpenAIClient
from lightrag.components.output_parsers.outputs import JsonOutputParser
from lightrag.core.base_data_class import DataClass

# Define the structured data models using dataclasses
@dataclass
class Entity(DataClass):
    name: str = field(metadata={"description": "The name of the entity."})
    type: str = field(metadata={"description": "The type of the entity (e.g., Person, Place, Concept, Theory)."})

@dataclass
class Relation(DataClass):
    source: str = field(metadata={"description": "The name of the source entity."})
    target: str = field(metadata={"description": "The name of the target entity."})
    relation: str = field(metadata={"description": "A descriptive label for the relationship."})

@dataclass
class KnowledgeGraph(DataClass):
    entities: List[Entity] = field(default_factory=list, metadata={"description": "A list of all unique entities found in the text."})
    relations: List[Relation] = field(default_factory=list, metadata={"description": "A list of relationships between the entities."})
    keywords: List[str] = field(default_factory=list, metadata={"description": "A list of key terms or topics in the text."})

# Implement the KnowledgeExtractor class
class KnowledgeExtractor:
    def __init__(self, llm_client: ModelClient, model_kwargs: dict):
        json_parser = JsonOutputParser(data_class=KnowledgeGraph, return_data_class=True)
        format_instructions = json_parser.format_instructions()

        prompt_template = (
            "You are a helpful assistant that extracts information into a knowledge graph.\n"
            "Your task is to analyze the provided text and extract entities, relations, and keywords.\n"
            "You MUST respond with ONLY a valid JSON object that conforms to the specified schema and nothing else. "
            "Do not add any conversational text, greetings, or explanations before or after the JSON object.\n"
            "The text to analyze is:\n"
            "--- TEXT START ---\n"
            "{text_to_analyze}\n"
            "--- TEXT END ---\n\n"
            f"{format_instructions}"
        )

        self.generator = Generator(
            model_client=llm_client,
            template=prompt_template,
            output_processors=json_parser,
            model_kwargs=model_kwargs
        )

    async def extract(self, text: str) -> Optional[KnowledgeGraph]:
        if not text or not text.strip():
            return None
        try:
            print(f"  - Extracting knowledge from segment: '{text[:50].strip()}...'")
            response = await self.generator.acall(prompt_kwargs={"text_to_analyze": text})
            return response.data
        except Exception as e:
            print(f"  - ERROR during knowledge extraction: {e}")
            return None

def create_knowledge_extractor(llm_provider: str = "deepseek") -> KnowledgeExtractor:
    """
    Initializes and returns a KnowledgeExtractor instance based on the specified provider.
    """
    if llm_provider == "alibaba":
        print("Using Alibaba Tongyi (qwen3-32b) model.")
        api_key = os.getenv("ALIBABA_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen3-32b"
        temperature = 0.7
    else: # Default to deepseek
        print("Using DeepSeek (deepseek-reasoner) model.")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-reasoner"
        temperature = 0.6

    if not api_key:
        raise ValueError(f"API key for {llm_provider} not found in .env file.")

    # Set environment variables for the lightrag OpenAIClient to use
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url

    llm_client = OpenAIClient(api_key=api_key) # api_key is still needed for the constructor

    model_kwargs = {"model": model, "temperature": temperature}

    # Add provider-specific parameters
    if llm_provider == "alibaba":
        model_kwargs["extra_body"] = {"enable_thinking": False}

    return KnowledgeExtractor(llm_client=llm_client, model_kwargs=model_kwargs)
