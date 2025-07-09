# This file will contain the functions that act as nodes in our LangGraph.
from typing import List, Dict, Any
# from .agent import EchartsAgentState # Causes circular import if EchartsAgentState is only in agent.py and nodes are imported there.
                                     # It's better to define EchartsAgentState in a separate file or pass dicts.
                                     # For simplicity with TypedDict, we might need to redefine or import carefully.
                                     # For now, let's assume state is a dictionary matching EchartsAgentState structure.

def parse_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the input JSON data and query.
    Currently normalizes input_json_data to a list of dictionaries if it's not already.
    """
    print("--- PARSING INPUT ---")
    input_json_data = state.get("input_json_data")
    query = state.get("query")

    parsed_data = None
    validation_errors = state.get("validation_errors", [])

    if isinstance(input_json_data, list):
        if all(isinstance(item, dict) for item in input_json_data):
            parsed_data = input_json_data
            print(f"Input data is a list of dictionaries. Parsed {len(parsed_data)} items.")
        else:
            err_msg = "Invalid input_json_data: If a list, all items must be dictionaries."
            print(err_msg)
            validation_errors.append(err_msg)
    # Basic support for dict of lists - assuming all lists are of the same length
    # and can be transformed into a list of dicts.
    # Example: {"categories": ["A", "B"], "values": [10, 20]} -> [{"categories": "A", "values": 10}, ...]
    elif isinstance(input_json_data, dict):
        print("Input data is a dictionary. Attempting to transform to list of dictionaries.")
        lists = [v for v in input_json_data.values() if isinstance(v, list)]
        if not lists:
            err_msg = "Invalid input_json_data: If a dictionary, it must contain lists of data."
            print(err_msg)
            validation_errors.append(err_msg)
        else:
            try:
                list_len = len(lists[0])
                if not all(len(l) == list_len for l in lists):
                    err_msg = "Invalid input_json_data: If a dictionary of lists, all lists must have the same length."
                    print(err_msg)
                    validation_errors.append(err_msg)
                else:
                    keys = list(input_json_data.keys())
                    parsed_data = []
                    for i in range(list_len):
                        item = {}
                        for key in keys:
                            if isinstance(input_json_data[key], list):
                                item[key] = input_json_data[key][i]
                            else: # if a scalar value is mixed in, replicate it (less common)
                                item[key] = input_json_data[key]
                        parsed_data.append(item)
                    print(f"Transformed dictionary of lists to {len(parsed_data)} items.")
            except Exception as e:
                err_msg = f"Error transforming dictionary of lists: {e}"
                print(err_msg)
                validation_errors.append(err_msg)
    else:
        err_msg = "Invalid input_json_data: Must be a list of dictionaries or a dictionary of lists."
        print(err_msg)
        validation_errors.append(err_msg)

    updated_state = state.copy()
    updated_state["parsed_data"] = parsed_data
    if validation_errors:
      # If we add errors here, we might want to stop processing early.
      # For now, just record them.
      updated_state["validation_errors"] = validation_errors
      print(f"Validation errors after parsing: {validation_errors}")


    print(f"Query received: {query}")
    # Potentially parse the query here for specific instructions if not done by an LLM node

    return updated_state

def select_chart_type_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selects the chart type based on the query (simple keyword matching for now).
    """
    print("--- SELECTING CHART TYPE ---")
    query = state.get("query", "").lower()
    selected_chart_type = None

    if "bar chart" in query or "bar graph" in query:
        selected_chart_type = "bar"
    elif "line chart" in query or "line graph" in query:
        selected_chart_type = "line"
    elif "pie chart" in query or "pie graph" in query:
        selected_chart_type = "pie"
    # Add more types as needed

    if not selected_chart_type:
        # Default or error
        print("Could not determine chart type from query. Defaulting to 'bar'.") # Or raise error/set validation error
        selected_chart_type = "bar" # Fallback, could also be an error state
        # validation_errors = state.get("validation_errors", [])
        # validation_errors.append("Chart type could not be determined from query.")
        # state["validation_errors"] = validation_errors


    print(f"Selected chart type: {selected_chart_type}")
    updated_state = state.copy()
    updated_state["selected_chart_type"] = selected_chart_type
    return updated_state

def _extract_keys_from_query(query: str, data_keys: List[str]) -> Dict[str, str]:
    """
    Very basic heuristic to extract category and value keys from a query.
    Example: "show sales per month" -> category_key: 'month', value_key: 'sales'
    """
    # This is a placeholder for more sophisticated NLP/LLM-based extraction
    query_parts = query.lower().split()
    mapping = {}

    # Common patterns: "Y per X [by Z]", "Y by X [by Z]", "Y over X [by Z]"
    # Z would be series_key, X category_key, Y value_key
    query_lower = query.lower()

    # Try to find series key first from "by Z" or "for each Z" patterns
    potential_series_keywords = ["product", "products", "type", "types", "group", "groups", "segment", "segments", "category"] # adding category here as it can be a series sometimes

    # Check for "by <series_keyword>" or "for each <series_keyword>"
    for sk_candidate_query in potential_series_keywords:
        if f"by {sk_candidate_query}" in query_lower or f"for each {sk_candidate_query}" in query_lower or f"all {sk_candidate_query}" in query_lower:
            for dk in data_keys:
                if sk_candidate_query.rstrip('s') in dk.lower():
                    mapping["series_key"] = dk
                    break
            if "series_key" in mapping:
                break

    # Try to find value and category keys using "per", "by" (as separator), "over"
    # Example: "sales per month"
    separator_keywords = ["per", "by", "over"] # "by" can be ambiguous (separator or series indicator)

    for sep in separator_keywords:
        if f" {sep} " in query_lower:
            parts = query_lower.split(f" {sep} ", 1)
            potential_value_str = parts[0].split()[-1] # last word before separator
            potential_category_str = parts[1].split()[0] # first word after separator

            for dk in data_keys:
                if not mapping.get("value_key") and (potential_value_str in dk.lower() or dk.lower() in potential_value_str):
                    mapping["value_key"] = dk
                if not mapping.get("category_key") and (potential_category_str in dk.lower() or dk.lower() in potential_category_str):
                    # Ensure category_key is not the same as an already identified series_key
                    if dk != mapping.get("series_key"):
                         mapping["category_key"] = dk

            if mapping.get("value_key") and mapping.get("category_key"):
                break # Found both based on separator

    # Fallback: if query mentions keys directly, and they haven't been filled by structured phrases
    for key in data_keys:
        if key.lower() in query_lower:
            if not mapping.get("value_key") and ("value" in key.lower() or "sales" in key.lower() or "count" in key.lower() or "amount" in key.lower()):
                mapping["value_key"] = key
            elif not mapping.get("category_key") and key != mapping.get("series_key") and \
                 ("month" in key.lower() or "date" in key.lower() or "day" in key.lower() or "year" in key.lower() or "name" in key.lower() or "label" in key.lower() or "item" in key.lower() or ("category" in key.lower() and "series_key" not in mapping) ): # if "category" is a key and not already series
                mapping["category_key"] = key
            elif not mapping.get("series_key") and key != mapping.get("value_key") and key != mapping.get("category_key") and \
                 ("product" in key.lower() or "type" in key.lower() or "group" in key.lower() or "segment" in key.lower() or ("category" in key.lower() and "category_key" not in mapping)): # if "category" is a key and not already category
                mapping["series_key"] = key

    # Clean up: ensure no key is assigned to multiple roles if alternatives exist or make sense
    if mapping.get("category_key") == mapping.get("series_key"):
        # This often happens if "category" is used as series. Try to find another category.
        # Or if the query was "sales by category" (where category is series) and another field is the actual x-axis category
        current_cat = mapping.get("category_key")
        for key in data_keys: # Try to find a more typical category key
            if key != current_cat and key != mapping.get("value_key"):
                 if ("month" in key.lower() or "date" in key.lower() or "time" in key.lower() or "name" in key.lower() or "label" in key.lower()):
                    mapping["category_key"] = key
                    break
        if mapping.get("category_key") == current_cat: # Still the same, means no better alternative found or needed.
             # If series_key is 'category' and category_key is also 'category', it implies we might not have a different category axis.
             # This could be fine for some charts (e.g. grouped bar where groups are categories, and x-axis is also derived from those categories)
             # or it might indicate an issue. For "sales by product over month", product=series, month=category, sales=value.
             # If "sales by category" and category is product type, then category=series, and we need an x-axis.
             # If the data is just [{"category": "A", "sales":100}, {"category": "B", "sales":200}], then category is category_key.
             pass


    return mapping


def map_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps the parsed data to a structure suitable for ECharts based on chart type and query.
    Populates 'data_mapping_config' and 'echarts_data_struct'.
    """
    print("--- MAPPING DATA ---")
    parsed_data = state.get("parsed_data")
    query = state.get("query", "").lower()
    chart_type = state.get("selected_chart_type")
    validation_errors = state.get("validation_errors", [])

    updated_state = state.copy()

    if not parsed_data:
        validation_errors.append("Cannot map data: parsed_data is missing.")
        updated_state["validation_errors"] = validation_errors
        print("Error: parsed_data is missing.")
        return updated_state

    # Get all available keys from the first data item
    data_keys = list(parsed_data[0].keys()) if parsed_data else []
    if not data_keys:
        validation_errors.append("Cannot map data: no keys found in parsed_data items.")
        updated_state["validation_errors"] = validation_errors
        print("Error: No keys found in parsed_data items.")
        return updated_state

    # 1. Infer Key Mappings (data_mapping_config)
    # This is a very basic heuristic. Could be an LLM call.
    # For "Create a bar chart showing values per category"
    # data = [{"category": "Mon", "value": 120, "type": "sales"}, ...]
    # We want category_key = "category", value_key = "value"

    mapping_config = _extract_keys_from_query(query, data_keys)

    # Fallback/Guessing if query extraction failed or wasn't specific enough
    if not mapping_config.get("category_key") and not mapping_config.get("value_key"):
        # Try to guess based on common names
        for key in data_keys:
            if key.lower() in ["category", "name", "label", "x", "product", "month", "day", "item"]:
                if not mapping_config.get("category_key"):
                    mapping_config["category_key"] = key
            elif key.lower() in ["value", "y", "count", "amount", "sales", "quantity", "score"]:
                if not mapping_config.get("value_key"):
                     mapping_config["value_key"] = key
            elif key.lower() in ["type", "group", "product", "segment"] : # Guess series_key
                if not mapping_config.get("series_key") and key != mapping_config.get("category_key") and key != mapping_config.get("value_key"):
                    mapping_config["series_key"] = key

    # Fallback guessing if specific keys are still missing
    # Guess category_key
    if not mapping_config.get("category_key") and parsed_data:
        for key in data_keys:
            if key != mapping_config.get("value_key") and key != mapping_config.get("series_key"):
                if isinstance(parsed_data[0].get(key), str) or "month" in key.lower() or "date" in key.lower() or "name" in key.lower() or "label" in key.lower() or "item" in key.lower():
                    mapping_config["category_key"] = key
                    break
        # If still no category key, pick first string key not used
        if not mapping_config.get("category_key"):
            for key in data_keys:
                if isinstance(parsed_data[0].get(key), str) and key != mapping_config.get("value_key") and key != mapping_config.get("series_key"):
                    mapping_config["category_key"] = key
                    break

    # Guess value_key
    if not mapping_config.get("value_key") and parsed_data: # Correctly aligned
        for key in data_keys:
            if key != mapping_config.get("category_key") and key != mapping_config.get("series_key"):
                if isinstance(parsed_data[0].get(key), (int, float)) or "value" in key.lower() or "sales" in key.lower():
                    mapping_config["value_key"] = key
                    break
        # If still no value key, pick first numeric key not used
        if not mapping_config.get("value_key"): # Correctly aligned with the inner if
            for key in data_keys:
                if isinstance(parsed_data[0].get(key), (int, float)) and key != mapping_config.get("category_key") and key != mapping_config.get("series_key"):
                    mapping_config["value_key"] = key
                    break

    # Guess series_key (less aggressive if others are set)
    if not mapping_config.get("series_key") and parsed_data: # Correctly aligned
        for key in data_keys:
            if key != mapping_config.get("category_key") and key != mapping_config.get("value_key"):
                # Typically, series keys are categorical strings
                if isinstance(parsed_data[0].get(key), str) and key.lower() in ["type", "group", "product", "segment", "category"]:
                     mapping_config["series_key"] = key
                     break

    print(f"Inferred mapping config: {mapping_config}")
    updated_state["data_mapping_config"] = mapping_config

    # 2. Transform Data (echarts_data_struct)
    # This part will be chart-type specific
    category_key = mapping_config.get("category_key")
    value_key = mapping_config.get("value_key")
    series_key = mapping_config.get("series_key") # For multi-series charts

    if not category_key or not value_key:
        err_msg = "Could not determine category_key or value_key for mapping."
        print(f"Error: {err_msg} - Available keys: {data_keys}, Query: {query}")
        validation_errors.append(err_msg)
        updated_state["validation_errors"] = validation_errors
        return updated_state

    echarts_data_struct = {}

    if chart_type in ["bar", "line"]:
        categories = sorted(list(set(d[category_key] for d in parsed_data if category_key in d)))

        # Handle single series vs multiple series
        if series_key and series_key in data_keys:
            series_names = sorted(list(set(d[series_key] for d in parsed_data if series_key in d)))
            echarts_series = []
            for s_name in series_names:
                series_data = []
                for cat in categories:
                    value = None
                    # Find the value for this category and series
                    # Assumes one data point per category/series combination.
                    # If multiple, could aggregate (e.g., sum, average) - requires more logic
                    for item in parsed_data:
                        if item.get(category_key) == cat and item.get(series_key) == s_name:
                            value = item.get(value_key)
                            break
                    series_data.append(value) # Append value or None if not found
                echarts_series.append({"name": s_name, "type": chart_type, "data": series_data})
            echarts_data_struct["series_names"] = series_names
        else: # Single series
            series_data = []
            for cat in categories:
                value = None
                # Find value for this category. If multiple items for a category, this picks the first.
                # Consider aggregation if necessary (e.g. sum if multiple entries for 'Mon')
                for item in parsed_data:
                    if item.get(category_key) == cat:
                        value = item.get(value_key)
                        break
                series_data.append(value)
            echarts_series = [{"name": value_key, "type": chart_type, "data": series_data}] # Use value_key as default series name

        echarts_data_struct["categories"] = categories
        echarts_data_struct["series"] = echarts_series

    elif chart_type == "pie":
        # For pie chart, data is typically list of {"name": category_value, "value": value_value}
        # If series_key is present, it might mean multiple pie charts or a more complex single pie.
        # For simplicity, let's assume a single pie chart for now.
        pie_data = []
        # If category_key and value_key are distinct, and no series_key or series_key is not for grouping slices
        # This assumes each row in parsed_data is a slice.
        # Example: [{"product": "A", "sales": 100}, {"product": "B", "sales": 150}]
        # Here, "product" is the name of the slice, "sales" is its value.
        for item in parsed_data:
            name = item.get(category_key)
            value = item.get(value_key)
            if name is not None and value is not None:
                pie_data.append({"name": name, "value": value})

        # Alternative: if data is like [{"category": "A", "value1": 10, "value2": 20}] and query means "pie of value1 vs value2 for category A"
        # This is more complex and not handled by the current simple key mapping.

        echarts_data_struct["series_data"] = pie_data # For a single pie series
        # The name of the series for a pie chart can often be derived from the query or be a default.

    else:
        validation_errors.append(f"Data mapping for chart type '{chart_type}' not implemented.")

    print(f"Generated ECharts data structure: {echarts_data_struct}")
    updated_state["echarts_data_struct"] = echarts_data_struct
    if validation_errors:
        updated_state["validation_errors"] = validation_errors

    return updated_state

def generate_echarts_options_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the ECharts options JSON structure based on the mapped data and chart type.
    """
    print("--- GENERATING ECHARTS OPTIONS ---")
    chart_type = state.get("selected_chart_type")
    echarts_data_struct = state.get("echarts_data_struct")
    # query = state.get("query") # Could be used for title
    data_mapping_config = state.get("data_mapping_config", {})
    validation_errors = state.get("validation_errors", [])

    updated_state = state.copy()

    if not chart_type or not echarts_data_struct:
        validation_errors.append("Cannot generate options: chart_type or echarts_data_struct is missing.")
        updated_state["validation_errors"] = validation_errors
        print("Error: chart_type or echarts_data_struct is missing.")
        return updated_state

    options = {
        "title": {"text": f"{chart_type.capitalize()} Chart"}, # Basic title
        "tooltip": {"trigger": "axis" if chart_type in ["bar", "line"] else "item"},
        "legend": {"data": echarts_data_struct.get("series_names", []) if chart_type in ["bar", "line"] else None},
        "series": []
    }

    if chart_type in ["bar", "line"]:
        options["xAxis"] = {
            "type": "category",
            "data": echarts_data_struct.get("categories", []),
            "name": data_mapping_config.get("category_key", "") # Use mapped category key as axis name
        }
        options["yAxis"] = {
            "type": "value",
            "name": data_mapping_config.get("value_key", "") # Use mapped value key as axis name
        }
        options["series"] = echarts_data_struct.get("series", [])
        if echarts_data_struct.get("series_names"):
            options["legend"]["data"] = echarts_data_struct.get("series_names")
        elif len(options["series"]) == 1 and options["series"][0].get("name"): # Single series, use its name for legend if not already covered
             options["legend"]["data"] = [options["series"][0].get("name")]


    elif chart_type == "pie":
        pie_series_data = echarts_data_struct.get("series_data", [])
        series_name = data_mapping_config.get("value_key", "Values") # Default name for pie series

        # Attempt to get a better series name from query or value_key
        query = state.get("query", "").lower()
        if "distribution of" in query:
            try:
                series_name = query.split("distribution of")[1].split("by")[0].strip().capitalize()
            except:
                pass
        elif data_mapping_config.get("value_key"):
             series_name = data_mapping_config.get("value_key").capitalize()


        options["series"] = [{
            "name": series_name,
            "type": "pie",
            "radius": "50%", # Default radius
            "data": pie_series_data,
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
        # Legend for pie chart usually comes from data item names
        options["legend"] = {"orient": "vertical", "left": "left", "data": [item["name"] for item in pie_series_data if "name" in item]}
        options["tooltip"]["trigger"] = "item"
        options["tooltip"]["formatter"] = "{a} <br/>{b} : {c} ({d}%)" # {a}: series name, {b}: data item name, {c}: value, {d}: percentage
    else:
        validation_errors.append(f"Option generation for chart type '{chart_type}' not implemented.")

    print(f"Generated ECharts options: {options}")
    updated_state["echarts_options"] = options
    if validation_errors:
        updated_state["validation_errors"] = validation_errors

    return updated_state

# Future node:
# def validate_options_node(state: Dict[str, Any]) -> Dict[str, Any]: ...
