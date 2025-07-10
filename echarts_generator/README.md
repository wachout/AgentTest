# Echarts Configuration Generator Agent

This project implements an agent that uses a Large Language Model (LLM) to convert unstructured data and a user query into an Echarts JSON configuration. The agent includes a self-correction mechanism to validate and refine the generated Echarts JSON.

## Project Structure

```
echarts_generator/
├── agents/
│   └── echarts_agent.py  # Contains the core EchartsAgent class
├── examples/
│   └── run_agent.py      # Example script to run the agent
├── tools/
│   └── (empty)           # Placeholder for any future tools
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

1.  **Clone the repository (if applicable) or download the files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    The agent uses the DeepSeek API. You need to provide your API key.
    *   **Directly in code (NOT RECOMMENDED for production):**
        Modify the `API_KEY` variable in `echarts_generator/agents/echarts_agent.py`.
        ```python
        API_KEY = "your_actual_deepseek_api_key"
        ```
    *   **Using Environment Variables (Recommended):**
        The agent is not currently set up to read from environment variables by default, but this is a good practice. You would typically modify `echarts_agent.py` to use `os.environ.get("DEEPSEEK_API_KEY")`.
        If you modify the code to use an environment variable named `DEEPSEEK_API_KEY`, you can set it in your shell:
        ```bash
        export DEEPSEEK_API_KEY="your_actual_deepseek_api_key"
        ```
        Or create a `.env` file in the `echarts_generator` directory with the following content:
        ```
        DEEPSEEK_API_KEY="your_actual_deepseek_api_key"
        ```
        And ensure your script loads it (e.g., using `python-dotenv`). The current example script does not automatically load `.env` files.

## Running the Agent

To run the example script:

```bash
python echarts_generator/examples/run_agent.py
```

This will execute the predefined examples in the `run_agent.py` script, print the input data, user query, and the generated Echarts JSON configuration.

## How it Works

1.  **Input:** The agent takes two main inputs:
    *   `data`: A Python dictionary containing the data to be visualized.
    *   `query`: A string describing what the user wants to visualize from the data.

2.  **Echarts Generation:**
    *   The `EchartsAgent` uses a Langchain chain with a specifically crafted prompt (`data_to_echarts_prompt`) to instruct the LLM (DeepSeek model) to convert the data and query into an Echarts JSON structure.
    *   It tries to ensure all metrics are included and appropriate chart types are chosen.

3.  **Validation and Self-Correction:**
    *   The initially generated JSON string is then passed to another Langchain chain with a `validation_prompt`. This asks the LLM to act as a validator for the Echarts JSON.
    *   If the validation feedback suggests issues, or if the generated string is not valid JSON, the feedback is appended to the original query, and the generation process is retried.
    *   This loop continues for a predefined number of retries (`max_retries` in `echarts_agent.py`).

4.  **Output:** If successful, the agent returns a string containing the Echarts JSON configuration.

## Key Components

*   **`echarts_generator/agents/echarts_agent.py`:**
    *   `EchartsAgent` class: Manages the interaction with the LLM.
    *   `llm`: Instance of `ChatOpenAI` configured for the DeepSeek model.
    *   `data_to_echarts_prompt`: Template for generating Echarts JSON.
    *   `validation_prompt`: Template for validating the generated JSON.
    *   `generate_echarts_config()`: Orchestrates the generation and validation process.
*   **Langchain:** The project uses Langchain for:
    *   LLM integration (`ChatOpenAI`).
    *   Prompt management (`ChatPromptTemplate`).
    *   Output parsing (`StrOutputParser`).
    *   Building processing chains (LCEL).

## Customization

*   **LLM Configuration:** Modify the `llm` instance in `echarts_agent.py` to change model parameters, temperature, or even the model provider (if compatible with `ChatOpenAI`).
*   **Prompts:** The core logic is heavily influenced by the prompts. You can refine `data_to_echarts_prompt` and `validation_prompt` in `echarts_agent.py` to improve accuracy or tailor the output.
*   **Data Handling:** The agent currently expects data as a Python dictionary and converts it to a string for the LLM. More sophisticated data preprocessing could be added.
*   **Error Handling:** The retry mechanism and error messages can be further enhanced.
```
