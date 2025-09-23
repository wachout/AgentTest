import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

def post_process_splits(splits):
    # Pass 1: Merge HTML blocks
    html_merged_splits = []
    i = 0
    while i < len(splits):
        current_split = splits[i]
        content = current_split.page_content

        # A simple heuristic for a split HTML block
        if "<html>" in content and "</html>" not in content:
            merged_content = content
            # Look ahead to find the closing tag
            for j in range(i + 1, len(splits)):
                next_content = splits[j].page_content
                merged_content += next_content
                if "</html>" in next_content:
                    html_merged_splits.append(Document(page_content=merged_content))
                    i = j # Move index past the merged splits
                    break
            else: # If loop finishes without finding a closing tag
                html_merged_splits.append(Document(page_content=merged_content))
        else:
            html_merged_splits.append(current_split)
        i += 1

    return html_merged_splits

def main():
    loader = DirectoryLoader('test_documents', glob="**/*.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    final_all_splits = []
    for doc in docs:
        splits = text_splitter.split_documents([doc])
        processed_splits = post_process_splits(splits)
        final_all_splits.extend(processed_splits)

    for i, split in enumerate(final_all_splits):
        print(f"--- Split {i} ---")
        print(split.page_content)
        print("\\n")

if __name__ == "__main__":
    main()
