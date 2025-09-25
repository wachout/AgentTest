import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

# Load environment variables from .env file
load_dotenv()

# Get Tongyi API key and base URL from environment variables
tongyi_api_key = os.getenv("TONGYI_API_KEY")
tongyi_base_url = os.getenv("TONGYI_BASE_URL")

# Check if the credentials are provided
if not tongyi_api_key or not tongyi_base_url:
    raise ValueError("TONGYI_API_KEY and TONGYI_BASE_URL must be set in the .env file")

# Initialize the language model
llm = ChatTongyi(
    temperature=0.7,
    model="qwen-plus", # Using qwen-plus for better performance
    api_key=tongyi_api_key,
    base_url=tongyi_base_url,
)

# Bind the model with a setting to disable thinking process if needed
# This can sometimes improve structured output generation
llm = llm.bind(enable_thinking=False)