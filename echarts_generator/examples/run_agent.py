import sys
import os
import json

# Add the parent directory to the Python path to allow importing the agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.echarts_agent import EchartsAgent, llm

def main():
    # Initialize the agent
    # The llm instance is imported from agents.echarts_agent
    agent = EchartsAgent(llm_instance=llm)

    print("--- Example 1: Complex Data ---")
    sample_data_1 = {
        "product_sales": [
            {"name": "Laptops", "value": 1200},
            {"name": "Phones", "value": 800},
            {"name": "Tablets", "value": 450}
        ],
        "monthly_visitors": [
            {"month": "Jan", "count": 10000},
            {"month": "Feb", "count": 12000},
            {"month": "Mar", "count": 15000}
        ],
        "category_distribution": {
            "Electronics": 0.6,
            "Clothing": 0.25,
            "Books": 0.15
        }
    }
    user_query_1 = "Show me the sales performance and monthly visitor trends. Also, I want to see the category distribution as a pie chart."

    try:
        print(f"User Query: {user_query_1}")
        print(f"Input Data: {json.dumps(sample_data_1, indent=2)}")
        echarts_config_str_1 = agent.generate_echarts_config(sample_data_1, user_query_1)
        print("\nGenerated Echarts Configuration (Example 1):")
        print(echarts_config_str_1)

        # Validate if the output is indeed a parsable JSON
        parsed_config_1 = json.loads(echarts_config_str_1)
        print("\nSuccessfully parsed generated Echarts JSON (Example 1). Output is valid JSON.")
        # You could pretty-print the parsed JSON here if desired
        # print(json.dumps(parsed_config_1, indent=2))

    except ValueError as e:
        print(f"\nError generating Echarts config (Example 1): {e}")
    except json.JSONDecodeError as e:
        print(f"\nGenerated output is not valid JSON (Example 1): {e}")
        print(f"Problematic string: {echarts_config_str_1}")
    except Exception as e:
        print(f"\nAn unexpected error occurred (Example 1): {e}")

    print("\n--- Example 2: Simpler Data ---")
    sample_data_2 = {
        "daily_temperatures": {"Monday": 22, "Tuesday": 24, "Wednesday": 23, "Thursday": 25, "Friday": 22}
    }
    user_query_2 = "Visualize the daily temperatures for this week as a line chart."
    try:
        print(f"User Query: {user_query_2}")
        print(f"Input Data: {json.dumps(sample_data_2, indent=2)}")
        echarts_config_str_2 = agent.generate_echarts_config(sample_data_2, user_query_2)
        print("\nGenerated Echarts Configuration (Example 2):")
        print(echarts_config_str_2)

        parsed_config_2 = json.loads(echarts_config_str_2)
        print("\nSuccessfully parsed generated Echarts JSON (Example 2). Output is valid JSON.")
        # print(json.dumps(parsed_config_2, indent=2))

    except ValueError as e:
        print(f"\nError generating Echarts config (Example 2): {e}")
    except json.JSONDecodeError as e:
        print(f"\nGenerated output is not valid JSON (Example 2): {e}")
        print(f"Problematic string: {echarts_config_str_2}")
    except Exception as e:
        print(f"\nAn unexpected error occurred (Example 2): {e}")

if __name__ == '__main__':
    # The agent (agents.echarts_agent) now handles API key loading from environment variables (.env file)
    # and prints a warning if the key is missing or a placeholder.
    # Ensure your DEEPSEEK_API_KEY is set in a .env file in the echarts_generator directory or in your environment.
    # Example .env file content:
    # DEEPSEEK_API_KEY="your_actual_deepseek_api_key"

    print("Attempting to run examples...")
    print("If you see API errors, please ensure your DEEPSEEK_API_KEY is correctly set.")
    main()
