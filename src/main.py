import os
import argparse
from txt_reader import read_txt_file
from docx_reader import read_docx_file
from pdf_reader import read_pdf_file

def main():
    parser = argparse.ArgumentParser(description="Read different file types and convert to Markdown.")
    parser.add_argument("file_path", help="Path to the input file.")
    args = parser.parse_args()

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    file_extension = os.path.splitext(file_path)[1].lower()
    base_name = os.path.basename(file_path)
    output_filename = os.path.splitext(base_name)[0] + ".md"
    output_path = os.path.join("output", output_filename)

    os.makedirs("output", exist_ok=True)

    if file_extension == ".txt":
        content = read_txt_file(file_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Content of {base_name} written to {output_path}")

    elif file_extension == ".docx":
        read_docx_file(file_path, output_path)
        print(f"Content of {base_name} converted to Markdown at {output_path}")

    elif file_extension == ".pdf":
        read_pdf_file(file_path, output_path)
        print(f"Content of {base_name} converted to Markdown at {output_path}")

    else:
        print(f"Error: Unsupported file type '{file_extension}'")

if __name__ == "__main__":
    main()
