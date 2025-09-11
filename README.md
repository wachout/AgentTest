# File Reader and Markdown Converter

This project is a Python application that reads `.txt`, `.docx`, and `.pdf` files. It converts `.docx` and `.pdf` files into Markdown format, extracting text, tables, and images.

## Features

- Reads plain text from `.txt` files.
- Converts `.docx` files to Markdown, including text, tables, and images.
- Converts `.pdf` files to Markdown, including text, tables, and images.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the application, run the `main.py` script from the command line, providing the path to the file you want to process.

```bash
python src/main.py <path-to-your-file>
```

### Examples

- **Processing a `.txt` file:**
  ```bash
  python src/main.py data/sample.txt
  ```
  The content of the `.txt` file will be saved in `output/sample.md`.

- **Processing a `.docx` file:**
  ```bash
  python src/main.py path/to/your/document.docx
  ```
  The converted Markdown file will be saved in `output/document.md`, and any images will be stored in `output/images/`.

- **Processing a `.pdf` file:**
  ```bash
  python src/main.py path/to/your/document.pdf
  ```
  The converted Markdown file will be saved in `output/document.md`, and any images will be stored in `output/images/`.

## Project Structure

```
.
├── data/
│   └── sample.txt
├── output/
├── src/
│   ├── main.py
│   ├── txt_reader.py
│   ├── docx_reader.py
│   └── pdf_reader.py
├── requirements.txt
└── README.md
```
