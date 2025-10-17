# LightRAG and LangGraph Agent

This project implements a sophisticated Retrieval-Augmented Generation (RAG) agent using LightRAG and LangGraph. The agent leverages a dual-structure index of vector embeddings and a knowledge graph to provide accurate and context-aware responses.

## Features

- **Dual-Structure Indexing**: Uses LightRAG to create both vector embeddings for semantic search and a knowledge graph for structured data retrieval.
- **Multi-Step Workflow**: Implements a multi-step agent workflow using LangGraph, including query rewriting, document retrieval, graph expansion, evidence reranking, and response generation.
- **"The Wandering Earth" Knowledge Base**: The agent is pre-loaded with knowledge from "The Wandering Earth" by Cixin Liu.

## Project Structure

```
.
├── data
│   └── sample.txt
├── src
│   └── agent.py
├── ingest.py
├── requirements.txt
└── testing_and_documentation.ipynb
```

- **`data/sample.txt`**: The source text from "The Wandering Earth".
- **`src/agent.py`**: The core LangGraph agent, including the state definition, tools, and graph workflow.
- **`ingest.py`**: The data ingestion script that processes the source text and creates the vector embeddings and knowledge graph.
- **`requirements.txt`**: The Python dependencies for the project.
- **`testing_and_documentation.ipynb`**: A Jupyter notebook for testing and documenting the agent's functionality.

## Getting Started

### Prerequisites

- Python 3.12+
- An OpenAI API key (optional, for query rewriting and response generation)
- A Neo4j instance (optional, for knowledge graph integration)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your environment variables by creating a `.env` file from the `.env.example` file and adding your API keys:
    ```bash
    cp .env.example .env
    ```

### Usage

1.  **Run the Jupyter Notebook**: The easiest way to get started is to run the `testing_and_documentation.ipynb` notebook. This will walk you through the entire process, from data ingestion to running the agent.

2.  **Run the Agent Directly**: You can also run the agent directly from the command line:
    ```bash
    python src/agent.py
    ```

## Knowledge Graph Integration (Future Work)

The current implementation includes placeholder nodes for knowledge graph integration. To enable this functionality, you will need to:

1.  Uncomment the Neo4j-related code in `ingest.py` and `src/agent.py`.
2.  Ensure you have a running Neo4j instance and that your credentials are correctly configured in your `.env` file.
3.  Implement the `graph_expander` tool in `src/agent.py` to query the knowledge graph and expand the retrieved documents with additional context.