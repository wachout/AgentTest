import sys
import json
from bs4 import BeautifulSoup

def convert_html_table_to_json(html_content):
    """
    Parses an HTML string, extracts the first table, and converts it to a JSON array of objects.
    The first row of the table is assumed to be the header.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return json.dumps([])

    headers = [header.text.strip() for header in table.find_all('th' if table.find('th') else 'td', recursive=False if table.find('th') else True)[:len(table.find('tr').find_all('td'))]]

    # If the first row was all `<td>`s, it will be duplicated, so we skip it
    rows_to_process = table.find_all('tr')[1:]

    data = []
    for row in rows_to_process:
        cells = row.find_all('td')
        if len(cells) == len(headers): # Ensure row is not malformed
            row_data = {headers[i]: cells[i].text.strip() for i in range(len(headers))}
            data.append(row_data)

    return json.dumps(data, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python html_to_json.py <path_to_html_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # The user's provided HTML is fragmented. We need to find all tables.
        # We'll use a simple split strategy for this case.
        html_parts = html.split('<html><body>')
        all_json_outputs = []

        for part in html_parts:
            if '<table>' in part:
                full_html = f"<html><body>{part}"
                json_output = convert_html_table_to_json(full_html)
                # We need to parse the json string back to a python object to merge them
                json_data = json.loads(json_output)
                if json_data:
                    all_json_outputs.extend(json_data)

        print(json.dumps(all_json_outputs, indent=4, ensure_ascii=False))

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
