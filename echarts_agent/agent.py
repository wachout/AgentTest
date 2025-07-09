# Main script to define and run the ECharts generation agent.
import json
from typing import List, Dict, TypedDict, Any, Optional

from langgraph.graph import StateGraph, END

# Import node functions
from .nodes import (
    parse_input_node,
    select_chart_type_node,
    map_data_node,
    generate_echarts_options_node,
    # validate_options_node # To be implemented
)

# Define the state for our graph
class EchartsAgentState(TypedDict):
    input_json_data: Any  # Raw input JSON (list of dicts or dict of lists)
    query: str  # User query

    parsed_data: Optional[List[Dict[str, Any]]] # Standardized data format (e.g., list of dicts)
    selected_chart_type: Optional[str] # e.g., 'bar', 'line', 'pie'

    # Configuration derived from query/data for mapping
    data_mapping_config: Optional[Dict[str, str]] # e.g., {"category_key": "category", "value_key": "value", "series_key": "type"}

    # Mapped data structured for ECharts generation
    # This might vary based on chart type. Example for simple charts:
    echarts_data_struct: Optional[Dict[str, Any]] # e.g. {"categories": [...], "series": [{"name": "s1", "data": [...]}]}

    echarts_options: Optional[Dict[str, Any]] # The generated ECharts options
    validation_errors: Optional[List[str]] # Errors found during validation
    feedback: Optional[str] # Feedback for refinement iterations

# Conditional function to check if parsing was successful
def should_continue_after_parsing(state: EchartsAgentState) -> str:
    """
    Determines the next step after parsing.
    If parsing errors occurred, it ends the process. Otherwise, continues to chart type selection.
    """
    if state.get("validation_errors") and any("Invalid input_json_data" in error for error in state["validation_errors"]):
        print("--- PARSING FAILED: Halting execution. ---")
        return END # Or a specific error handling node
    return "select_chart_type"

def main(input_json_string: str, query: str):
    """
    Runs the ECharts agent.
    """
    try:
        raw_input_data = json.loads(input_json_string)
    except json.JSONDecodeError:
        print("Error: Invalid input JSON string.")
        return

    # Initialize the graph
    workflow = StateGraph(EchartsAgentState)

    # Add nodes to the workflow
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("select_chart_type", select_chart_type_node)
    workflow.add_node("map_data", map_data_node)
    workflow.add_node("generate_options", generate_echarts_options_node)
    # workflow.add_node("validate_options", validate_options_node) # To be implemented

    # Define edges for the workflow
    workflow.set_entry_point("parse_input")

    # Conditional edge after parsing
    workflow.add_conditional_edges(
        "parse_input",
        should_continue_after_parsing,
        {
            "select_chart_type": "select_chart_type",
            END: END
        }
    )

    # For now, we'll end after chart type selection. # Old comment
    # Later, this will go to map_data node. # Old comment
    # workflow.add_edge("select_chart_type", END) # Old edge
    workflow.add_edge("select_chart_type", "map_data")
    workflow.add_edge("map_data", "generate_options")

    # For now, end after generate_options. Later, this goes to validate_options.
    workflow.add_edge("generate_options", END)
    # workflow.add_edge("generate_options", "validate_options")


    # ... more edges ...
    # workflow.add_conditional_edges( # Example for later
    #     "validate_options", # Example for later
    #     lambda state: "refine" if state.get("validation_errors") else END,
    #     {"refine": "map_data", END: END}
    # )

    # Compile the graph
    app = workflow.compile()

    # Prepare initial state
    initial_state = {
        "input_json_data": raw_input_data,
        "query": query,
        "parsed_data": None,
        "selected_chart_type": None,
        "data_mapping_config": None,
        "echarts_data_struct": None,
        "echarts_options": None,
        "validation_errors": [], # Initialize as empty list
        "feedback": None,
    }

    print(f"\n--- RUNNING AGENT FOR QUERY: '{query}' ---")
    # Run the graph
    final_state = app.invoke(initial_state)

    print("\n--- AGENT EXECUTION FINISHED ---")
    print("Final State:")
    # print(f"  Input JSON: {json.dumps(final_state.get('input_json_data'), indent=2)}") # Can be verbose
    # print(f"  Query: {final_state.get('query')}")
    print(f"  Parsed Data: {json.dumps(final_state.get('parsed_data'), indent=2)}")
    print(f"  Selected Chart Type: {final_state.get('selected_chart_type')}")
    print(f"  Data Mapping Config: {final_state.get('data_mapping_config')}")
    # print(f"  ECharts Data Structure: {json.dumps(final_state.get('echarts_data_struct'), indent=2)}") # Can be verbose
    print(f"\n  Generated ECharts Options:")
    print(json.dumps(final_state.get("echarts_options"), indent=2))

    if final_state.get("validation_errors"):
        print("\n  Validation Errors:")
        for error in final_state["validation_errors"]:
            print(f"    - {error}")

if __name__ == "__main__":
    # Example Usage
    sample_json_data_str = """
    [
      {"category": "Mon", "value": 120, "type": "sales"},
      {"category": "Tue", "value": 200, "type": "sales"},
      {"category": "Wed", "value": 150, "type": "sales"},
      {"category": "Thu", "value": 80, "type": "sales"},
      {"category": "Fri", "value": 70, "type": "sales"},
      {"category": "Sat", "value": 110, "type": "sales"},
      {"category": "Sun", "value": 130, "type": "sales"}
    ]
    """
    sample_query = "Create a bar chart showing values per category"

    main(sample_json_data_str, sample_query)

    sample_json_data_list_of_dicts_str = """
    [
        {"product": "Shirts", "sales": 1200, "month": "Jan"},
        {"product": "Pants", "sales": 800, "month": "Jan"},
        {"product": "Shoes", "sales": 1500, "month": "Jan"},
        {"product": "Shirts", "sales": 1300, "month": "Feb"},
        {"product": "Pants", "sales": 900, "month": "Feb"},
        {"product": "Shoes", "sales": 1600, "month": "Feb"}
    ]
    """
    sample_query_line = "Show monthly sales trend for all products as a line chart"
    main(sample_json_data_list_of_dicts_str, sample_query_line)

    sample_json_pie_data_str = """
    [
        {"category": "Electronics", "revenue": 50000},
        {"category": "Books", "revenue": 30000},
        {"category": "Clothing", "revenue": 45000},
        {"category": "Home Goods", "revenue": 25000}
    ]
    """
    sample_query_pie = "Display revenue distribution by category in a pie chart"
    main(sample_json_pie_data_str, sample_query_pie)

    # Example for a dictionary of lists (testing parsing and simple chart)
    sample_json_dict_of_lists_str = """
    {
      "item_name": ["Apples", "Bananas", "Cherries"],
      "quantity_sold": [150, 200, 120]
    }
    """
    sample_query_dict_bar = "Bar chart of quantity_sold per item_name"
    main(sample_json_dict_of_lists_str, sample_query_dict_bar)
