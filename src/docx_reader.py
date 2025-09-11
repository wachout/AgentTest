import os
from docx2python import docx2python
from docx2python.iterators import enum_at_depth

def read_docx_file(file_path, output_path):
    """
    Reads a .docx file, extracts text, tables, and images, and saves it as a Markdown file.

    Args:
        file_path (str): The path to the .docx file.
        output_path (str): The path to save the output Markdown file and images.
    """
    image_folder = os.path.join(os.path.dirname(output_path), 'images')
    os.makedirs(image_folder, exist_ok=True)

    with docx2python(file_path, image_folder=image_folder) as docx_content:
        with open(output_path, 'w', encoding='utf-8') as md_file:
            for (i, (table_type, table)) in enum_at_depth(docx_content.body, 1):
                if table_type == "table":
                    md_file.write(f"### Table {i+1}\n\n")
                    # Transpose the table
                    transposed_table = list(map(list, zip(*table)))

                    # Write header
                    header = " | ".join(str(cell).replace("\n", " ") for cell in transposed_table[0])
                    md_file.write(f"| {header} |\n")

                    # Write separator
                    separator = " | ".join(["---"] * len(transposed_table[0]))
                    md_file.write(f"| {separator} |\n")

                    # Write rows
                    for row in transposed_table[1:]:
                        row_content = " | ".join(str(cell).replace("\n", " ") for cell in row)
                        md_file.write(f"| {row_content} |\n")
                    md_file.write("\n")
                elif table_type == "paragraph":
                     md_file.write(table)
                elif table_type == "image":
                    image_name = os.path.basename(table)
                    md_file.write(f"![{image_name}](images/{image_name})\n\n")
