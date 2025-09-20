import sys
import json
import re
from bs4 import BeautifulSoup

def convert_html_table_to_json(html_content):
    """
    Parses an HTML string, extracts the first table, and converts it to a list of dicts.
    The first row of the table is assumed to be the header.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    if not rows:
        return []

    headers = [header.text.strip() for header in rows[0].find_all('td')]

    data = []
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) == len(headers):
            row_data = {headers[i]: cells[i].text.strip() for i in range(len(headers))}
            data.append(row_data)

    return data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python html_to_json.py <path_to_text_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_tables_data = []

        def replacer(match):
            """
            This function is called for each <html>...</html> match.
            It converts the table to JSON, adds it to our master list,
            and returns the JSON string for replacement.
            """
            html_block = match.group(0)
            table_data = convert_html_table_to_json(html_block)
            if table_data:
                all_tables_data.extend(table_data)
                # Return a compact JSON string for in-place replacement
                return json.dumps(table_data, ensure_ascii=False)
            # If no table is found, return the original block
            return html_block

        # Use re.sub with our replacer function to get the modified text
        modified_text = re.sub('<html>.*?</html>', replacer, content, flags=re.DOTALL)

        # Prepare the final structured output
        final_output = {
            "combined_json_tables": all_tables_data,
            "modified_text": modified_text
        }

        # Print the final JSON object
        print(json.dumps(final_output, indent=4, ensure_ascii=False))

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
