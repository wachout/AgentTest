import os
import requests
from urllib.parse import unquote, urlparse

def download_file(url, dest_folder):
    """
    Downloads a file from a URL and saves it to a destination folder.

    Args:
        url (str): The URL of the file to download.
        dest_folder (str): The path to the folder where the file should be saved.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        print(f"Destination folder '{dest_folder}' does not exist. Creating it.")
        os.makedirs(dest_folder)

    try:
        # Parse the URL to get the path component and decode it
        path = urlparse(url).path
        # The unquote function decodes URL-encoded characters (e.g., %E9 -> é)
        filename = unquote(os.path.basename(path))

        # If filename is empty, it means the URL might not have a clear file path
        if not filename:
            print("Error: Could not determine filename from URL.")
            return

        # Construct the full file path
        file_path = os.path.join(dest_folder, filename)
        print(f"Downloading '{filename}' from '{url}'...")

        # Make a request to the URL
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Write the content to the file in chunks
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Successfully saved file to '{file_path}'")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except IOError as e:
        print(f"Error saving file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Example Usage ---
    # NOTE: The agent's environment cannot access local network URLs like '192.168.x.x'.
    # This example is provided for the user to run in their own environment.

    example_url = "http://192.168.35.125:8500/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3/%E5%AE%89%E5%85%A8%E7%94%9F%E4%BA%A7/%E5%B7%A5%E4%B8%9A%E4%BA%92%E8%81%94%E7%BD%91%E5%8D%B1%E5%8C%96%E5%AE%89%E5%85%A8%E7%94%9F%E4%BA%A7%E5%BB%BA%E8%AE%BE%E6%A0%87%E5%87%86%E7%AC%AC3%E9%83%A8%E5%88%86%E4%BA%BA%E5%91%98%E5%AE%9A%E4%BD%8D.docx"
    destination_directory = "downfile"

    print("--- Running File Downloader Script ---")
    # To run this, you would call the function like this:
    # download_file(example_url, destination_directory)

    print("\nScript finished. To use it, uncomment the function call above and run:")
    print(f"python file_downloader.py")
    print("\nPlease ensure the URL is accessible from your machine.")