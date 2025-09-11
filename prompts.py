from langchain_core.prompts import PromptTemplate

# Prompt for the Chapter Splitter Agent
# This prompt asks the LLM to act as a text structure analyzer to find chapter titles.
# It specifies the output format as a JSON list of integers representing the start indices.
CHAPTER_SPLITTER_PROMPT = PromptTemplate.from_template(
    """
You are an expert in analyzing text structure. Your task is to identify the starting character index of each chapter in the following text.
A chapter usually starts with a title like "Chapter X", "第一章", "Section X", etc.

Scan the text and find the exact starting character position of every chapter title.

Return your findings as a JSON object with a single key "indices" containing a list of integer indices.
The first chapter always starts at index 0.

Example:
Text: "Chapter 1: The Beginning. Some text... Chapter 2: The Next Step. More text..."
Output: {{"indices": [0, 31]}}

Now, analyze the following text:

--- TEXT START ---
{input_text}
--- TEXT END ---

Your JSON output:
"""
)

# Prompt for the Paragraph Splitter Agent
# This prompt guides the LLM to find the beginning of each paragraph.
# It's designed to be more robust than a simple newline split, by understanding logical text blocks.
PARAGRAPH_SPLITTER_PROMPT = PromptTemplate.from_template(
    """
You are an expert in text formatting and layout. Your task is to identify the starting character index of every paragraph in the provided text.
A new paragraph is typically separated by one or more blank lines, but can also be a logical separation of ideas.
The very first paragraph starts at index 0.

Scan the entire text and provide the starting character index for each paragraph.

Return your findings as a JSON object with a single key "indices" containing a list of integer indices.

Example:
Text: "This is the first paragraph.\n\nThis is the second."
Output: {{"indices": [0, 29]}}

Now, analyze the following text:

--- TEXT START ---
{input_text}
--- TEXT END ---

Your JSON output:
"""
)

# Prompt for the Semantic Splitter Agent
# This is the most sophisticated prompt. It asks the LLM to perform a deep semantic analysis of the text.
# It needs to understand the flow of topics and identify points where the main idea shifts significantly.
SEMANTIC_SPLITTER_PROMPT = PromptTemplate.from_template(
    """
You are a highly intelligent AI with deep expertise in semantic analysis and topic modeling.
Your mission is to read the following text and identify the major thematic or topic shifts.
A semantic split should occur at the point where the text transitions from one main idea to another.

Think of it as identifying the natural breaking points for creating a summary or a table of contents based on meaning, not just formatting.
The first semantic section always starts at index 0.

Carefully read the text, understand its meaning and flow, and then provide the starting character indices of each distinct semantic section.
Return your findings as a JSON object with a single key "indices" containing a list of integer indices.

Now, analyze the following text:

--- TEXT START ---
{input_text}
--- TEXT END ---

Your JSON output:
"""
)
