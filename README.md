# Knowledge Graph Extraction from Text

This project demonstrates a pipeline for extracting a knowledge graph from a given text. It uses a Large Language Model (LLM) to perform the extraction of entities, keywords, and their relationships. The extracted information is then used to construct a graph using the NetworkX library, which is saved in a format that can be easily visualized or imported into a graph database.

## Features

- **Entity Extraction**: Identifies named entities (e.g., people, places, concepts).
- **Keyword Extraction**: Pulls out the most relevant keywords from the text.
- **Relationship Extraction**: Determines the connections between the identified entities.
- **Graph Construction**: Builds a directed graph from the extracted knowledge.
- **Persistent Storage**: Saves the graph to a `.graphml` file, which can be visualized with tools like [Gephi](https://gephi.org/) or imported into a graph database.

## How to Use

### 1. Setup

First, install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

**Note**: The LLM API key is currently hardcoded in `knowledge_extractor.py` for demonstration purposes. In a production environment, it is strongly recommended to use environment variables to manage sensitive keys.

### 2. Prepare Input Text

Place the text you want to analyze into a file named `wandering_earth_summary.txt`. For this demo, a summary of the plot of "The Wandering Earth" is used.

### 3. Run the Pipeline

Execute the main script to run the complete pipeline:

```bash
python main.py
```

The script will:
1. Read the input text.
2. Call the LLM to extract knowledge.
3. Save the structured knowledge to `knowledge.json`.
4. Build a graph from the knowledge.
5. Save the final graph to `knowledge_graph.graphml`.

### 4. View the Output

- **`knowledge.json`**: A JSON file containing the extracted entities, keywords, and relationships. This is the raw structured output from the LLM.
- **`knowledge_graph.graphml`**: An XML-based file format for graphs. You can open this file in graph visualization software like Gephi to explore the network of entities and their relationships.

## Project Structure

- `main.py`: The main script that orchestrates the entire pipeline.
- `knowledge_extractor.py`: Contains the logic for interacting with the LLM to extract information from the text.
- `graph_builder.py`: Contains the logic for building a NetworkX graph from the extracted knowledge.
- `wandering_earth_summary.txt`: The input text file.
- `requirements.txt`: A list of Python dependencies for the project.
- `README.md`: This file.