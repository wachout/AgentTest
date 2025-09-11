import os
import fitz  # PyMuPDF
from PIL import Image

def read_pdf_file(file_path, output_path):
    """
    Reads a .pdf file, extracts text, tables, and images, and saves it as a Markdown file.

    Args:
        file_path (str): The path to the .pdf file.
        output_path (str): The path to save the output Markdown file and images.
    """
    image_folder = os.path.join(os.path.dirname(output_path), 'images')
    os.makedirs(image_folder, exist_ok=True)

    doc = fitz.open(file_path)
    with open(output_path, 'w', encoding='utf-8') as md_file:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text
            text = page.get_text("text")
            md_file.write(text)
            md_file.write("\n\n")

            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"image_{page_num+1}_{img_index+1}.{image_ext}"
                image_path = os.path.join(image_folder, image_name)

                with open(image_path, 'wb') as img_file:
                    img_file.write(image_bytes)

                md_file.write(f"![{image_name}](images/{image_name})\n\n")

            # Extract tables
            tables = page.find_tables()
            if tables:
                md_file.write(f"### Tables on Page {page_num+1}\n\n")
                for i, table in enumerate(tables):
                    md_file.write(f"#### Table {i+1}\n\n")
                    df = table.to_pandas()
                    md_file.write(df.to_markdown(index=False))
                    md_file.write("\n\n")

    doc.close()
