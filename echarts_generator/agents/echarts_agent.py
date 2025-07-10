import os
import json # Added for json.dumps
from dotenv import load_dotenv # For environment variables
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Configure the LLM
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") # Default base URL

if not API_KEY:
    print("WARNING: DEEPSEEK_API_KEY environment variable not found.")
    print("Please set it in your environment or in a .env file.")
    # You might want to raise an error here or use a placeholder if absolutely necessary for some tests,
    # but for actual operation, the key is crucial.
    # For this exercise, we'll allow it to proceed, but API calls will fail.
    API_KEY = "sk-xxxx" # Placeholder to allow script to run, but expect API errors

llm = ChatOpenAI(
    temperature=0.6,
    model="deepseek-reasoner", # Ensure this model name is correct for DeepSeek
    api_key=API_KEY,
    base_url=BASE_URL,
)

class EchartsAgent:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.data_to_echarts_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in Echarts data visualization.
            Your task is to convert the given unstructured data (in JSON string format) and user query into a valid Echarts JSON configuration.

            User Query: {query}
            Input Data (JSON String): {data}

            Instructions:
            1. Analyze the input data (provided as a JSON string) and the user's query.
            2. Determine the most suitable Echarts chart type(s) for each metric in the data. Consider the query for context.
            3. Ensure ALL metrics from the input data are represented in the Echarts configuration.
            4. Generate a complete and valid Echarts JSON configuration.
            5. If multiple chart types are suitable, you can create a chart with multiple series or suggest options.
            6. Pay attention to data types and structure them correctly for Echarts (e.g., series, xAxis, yAxis, legend, tooltip).
            7. Your response MUST be only the Echarts JSON configuration, with no additional text, explanations, or markdown formatting (e.g. no ```json ... ```).

            Example of a simple Echarts structure for a bar chart:
            {{
              "title": {{ "text": "Sales Data" }},
              "tooltip": {{}},
              "legend": {{ "data": ["Sales"] }},
              "xAxis": {{ "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] }},
              "yAxis": {{}},
              "series": [{{ "name": "Sales", "type": "bar", "data": [120, 200, 150, 80, 70, 110, 130] }}]
            }}

            Echarts JSON Configuration (strictly JSON, no other text):
            """
        )

        self.validation_prompt = ChatPromptTemplate.from_template(
            """
            You are an Echarts configuration validator.
            Review the following Echarts JSON configuration.

            Echarts JSON:
            {echarts_json}

            Validation Checklist:
            1. Is the JSON valid?
            2. Does it adhere to the Echarts schema?
            3. Are all necessary components for a basic chart present (e.g., series, axis if applicable)?
            4. Does the chart type seem appropriate for the kind of data usually represented in such a structure?
            5. Are there any obvious errors or omissions?

            Your Feedback:
            If the JSON is valid, complete, and adheres to Echarts best practices, respond with the single word "VALID" and nothing else.
            Otherwise, provide a concise list of issues, each on a new line, starting with "INVALID:". Example:
            INVALID:
            - Missing yAxis configuration.
            - Series data for 'sales' is empty.
            """
        )

        self.echarts_chain = (
            {"data": RunnablePassthrough(), "query": RunnablePassthrough()}
            | self.data_to_echarts_prompt
            | self.llm
            | StrOutputParser()
        )

        self.validation_chain = (
            self.validation_prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_echarts_config(self, data: dict, query: str) -> str:
        """
        Generates an Echarts configuration from data and a query.
        Includes a self-correction step.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Step 1: Generate Echarts JSON
                generated_json_str = self.echarts_chain.invoke({"data": str(data), "query": query})

                # Basic check if the output is likely JSON (starts with { and ends with })
                # The LLM might sometimes return explanations before or after the JSON block.
                # We try to extract the JSON part.
                json_start = generated_json_str.find('{')
                json_end = generated_json_str.rfind('}')
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    generated_json_str = generated_json_str[json_start : json_end+1]
                else:
                    print(f"Attempt {attempt+1}: LLM did not return a JSON-like structure. Output: {generated_json_str}")
                    if attempt == max_retries - 1:
                         raise ValueError("LLM failed to generate a JSON-like structure after multiple attempts.")
                    # Potentially add the raw output to the next prompt for correction
                    # For now, we just retry with the original query.
                    query += f"\nPrevious attempt failed to produce valid JSON. Please ensure the output is strictly a JSON object. Previous output: {generated_json_str}"
                    continue # Retry generation


                # Step 2: Validate the generated JSON
                validation_feedback = self.validation_chain.invoke({"echarts_json": generated_json_str})
                print(f"Attempt {attempt+1} - Validation Feedback: {validation_feedback}")

                if "VALID" in validation_feedback.upper(): # Make check case-insensitive
                    # Further parse to ensure it's actual JSON
                    import json
                    json.loads(generated_json_str) # This will raise an error if not valid JSON
                    return generated_json_str
                else:
                    # If not valid, refine the query for the next attempt
                    query += f"\nReview Feedback: {validation_feedback}. Please address these issues in the next generation."
                    print(f"Attempt {attempt+1} failed validation. Refining query and retrying.")

            except json.JSONDecodeError as e:
                print(f"Attempt {attempt+1}: Generated Echarts config is not valid JSON: {e}")
                query += f"\nPrevious attempt resulted in a JSONDecodeError: {e}. Ensure the output is valid JSON. Problematic JSON: {generated_json_str}"
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid Echarts JSON after {max_retries} attempts. Last error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during generation or validation: {e}")
                if attempt == max_retries - 1:
                    raise e # Re-raise the last exception if all retries fail
                # Add error information to the query for the next attempt
                query += f"\nAn error occurred: {str(e)}. Please try to fix this in the next attempt."


        raise ValueError(f"Failed to generate a valid Echarts configuration after {max_retries} attempts.")

# The __main__ block has been moved to examples/run_agent.py
# You can run that script to see the agent in action.
