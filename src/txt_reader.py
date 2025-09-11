def read_txt_file(file_path):
    """
    Reads a .txt file and returns its content.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
