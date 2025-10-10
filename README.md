# Graph Extractor from Text using LightRAG and LangChain

This project provides a Python script (`graph_extractor.py`) to extract a knowledge graph (nodes and edges) from a given piece of text. It uses a Large Language Model (LLM) for the extraction and leverages the `lightrag` library for data structuring and parsing, and `langchain` for model interaction.

## Features

- **Reusable Function**: A core `extract_graph(text, llm)` function that can be easily integrated into other projects.
- **Secure API Key Management**: Uses a `.env` file to securely manage your LLM API keys, keeping them out of the source code.
- **Flexible LLM Integration**: Works with any LangChain-compatible chat model (e.g., `ChatOpenAI`, `ChatTongyi`).
- **Structured Output**: Uses `lightrag`'s data classes to ensure the output is a well-defined `KnowledgeGraph` object.

## Setup and Usage

Follow these steps to set up and run the project.

### 1. Install Dependencies

You need to install `lightrag`, `langchain-openai`, and `python-dotenv`. You can install them all using pip:

```bash
pip install lightrag langchain-openai python-dotenv
```

### 2. Set Up Environment Variables

For security, the script loads your API key from an environment file.

1.  Create a new file named `.env` in the root of the project directory.
2.  Add your API credentials to the `.env` file. For example, if you are using the DeepSeek model, your file should look like this:

    ```env
    DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
    ```

    Replace `sk-xxxxxxxxxxxxxxxxxxxxxxxx` with your actual API key. The script will automatically load these variables.

### 3. Run the Script

Once the dependencies are installed and the `.env` file is configured, you can run the script directly to see a working example:

```bash
python3 graph_extractor.py
```

The script will then:
1.  Load the API key from your `.env` file.
2.  Instantiate the `ChatOpenAI` model.
3.  Process a sample text.
4.  Print the final, structured knowledge graph to the console in JSON format.

## Integrating into Your Own Project

To use the graph extraction functionality in your own code, simply import the `extract_graph` function and the necessary data classes from the `graph_extractor.py` file.

```python
from graph_extractor import extract_graph, KnowledgeGraph, Node, Edge
from langchain_openai import ChatOpenAI

# 1. Instantiate your desired LLM
my_llm = ChatOpenAI(
    # ... your model configuration ...
)

# 2. Your text to process
my_text = "Your text containing entities and relationships goes here."

# 3. Call the function
result: KnowledgeGraph = extract_graph(text=my_text, llm=my_llm)

# 4. Use the result
for node in result.nodes:
    print(f"Found Node: {node.id} (Type: {node.type})")

for edge in result.edges:
    print(f"Found Edge: {edge.source} -> {edge.label} -> {edge.target}")

```