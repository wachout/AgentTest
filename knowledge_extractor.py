import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# It's good practice to use environment variables for API keys,
# but for this demonstration, we'll use the provided one directly.
# In a real application, you would use something like:
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("DEEPSEEK_API_KEY")

api_key = "sk-afce12abb3b142c787e85f8ec97c1ad2" # Example key from user
base_url = "https://api.deepseek.com/v1"

def extract_knowledge(text: str) -> dict:
    """
    Extracts entities, keywords, and relationships from a given text using an LLM.

    Args:
        text: The input text to process.

    Returns:
        A dictionary containing the extracted knowledge, structured as JSON.
    """
    llm = ChatOpenAI(
        temperature=0.6,
        model="deepseek-reasoner",
        api_key=api_key,
        base_url=base_url,
    )

    prompt_template = """
    From the text below, extract the following information:
    1.  **Entities**: Identify all named entities (people, locations, organizations, concepts, etc.). For each entity, provide its type (e.g., Person, Planet, Organization) and any relevant attributes.
    2.  **Keywords**: Extract the main keywords that summarize the text.
    3.  **Relationships**: Identify the relationships between the entities.

    Provide the output in a single valid JSON object with the following keys: "entities", "keywords", "relationships".

    - The "entities" value should be a list of objects, where each object has "name", "type", and "attributes" (a dictionary of key-value pairs).
    - The "keywords" value should be a list of strings.
    - The "relationships" value should be a list of objects, where each object has "source", "target", and "relation" (a string describing the connection).

    Make sure the source and target in relationships exactly match the names of the entities identified.

    Text to process:
    ---
    {text}
    ---
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm

    response = chain.invoke({"text": text})

    # The response content should be a JSON string, so we parse it.
    try:
        # The output from the LLM is in the `content` attribute of the message
        return json.loads(response.content)
    except json.JSONDecodeError:
        print("Error: The LLM did not return a valid JSON object.")
        print("LLM Output:\n", response.content)
        return None

if __name__ == "__main__":
    try:
        with open("wandering_earth_summary.txt", "r", encoding="utf-8") as f:
            summary_text = f.read()

        knowledge = extract_knowledge(summary_text)

        if knowledge:
            # Save the extracted knowledge to a JSON file for inspection
            with open("knowledge.json", "w", encoding="utf-8") as f:
                json.dump(knowledge, f, indent=4, ensure_ascii=False)
            print("Successfully extracted knowledge and saved it to knowledge.json")
            # Also print to console for immediate feedback
            print("\n--- Extracted Knowledge ---")
            print(json.dumps(knowledge, indent=2, ensure_ascii=False))
            print("-------------------------")

    except FileNotFoundError:
        print("Error: 'wandering_earth_summary.txt' not found. Please run the previous step first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")