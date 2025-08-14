import os
import re
import json
from typing import TypedDict, List
import warnings

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine, inspect, text, exc

# --- 1. CONFIGURATION ---
# Select your LLM provider and SQL dialect here
# llm_provider: "openai", "deepseek", or "tongyi"
# sql_dialect: "mysql" or "postgresql"
CONFIG = {
    "llm_provider": "openai",
    "sql_dialect": "sqlite",
    # IMPORTANT: Replace with your actual database connection string
    # Example for PostgreSQL: "postgresql://user:password@host:port/database"
    # Example for MySQL: "mysql+pymysql://user:password@host:port/database"
    "database_url": "sqlite:///:memory:", # Default to in-memory for safe, isolated testing
    "llm_configs": {
        "openai": {
            "api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "temperature": 0,
        },
        "deepseek": {
            "api_key": "sk-xxxx", # Placeholder
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-reasoner",
            "temperature": 0.6,
        },
        "tongyi": {
            "api_key": "sk-xxxx", # Placeholder
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen2-72b-instruct", # Using a recommended model
            "temperature": 0.7,
        },
    }
}

# --- 2. LLM and DATABASE FACTORY ---

def get_llm() -> BaseChatModel:
    """Initializes the correct LLM based on the global CONFIG."""
    provider = CONFIG["llm_provider"]
    config = CONFIG["llm_configs"][provider]

    if provider == "tongyi":
        try:
            # Note: Dashscope uses a specific parameter for the API key
            llm = ChatTongyi(
                model=config["model"],
                temperature=config["temperature"],
                dashscope_api_key=config["api_key"],
                base_url=config["base_url"],
            )
            # Example of binding model-specific parameters
            return llm.bind(enable_thinking=False)
        except ImportError:
            raise ImportError("Tongyi (Dashscope) requires `langchain-community` and `dashscope`. Please install with `pip install langchain-community dashscope`.")
    else: # Handles "openai" and "deepseek" as they share the ChatOpenAI class
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

def get_engine():
    """Creates a SQLAlchemy engine from the database_url in the CONFIG."""
    try:
        return create_engine(CONFIG["database_url"])
    except Exception as e:
        print(f"Error creating database engine: {e}")
        return None

# Initialize components
llm = get_llm()
engine = get_engine()

# --- 3. TOOLS / DATABASE HELPERS ---

def get_schema(eng):
    """Returns the database schema for the given engine."""
    if not eng: return "Error: Database engine not available."
    try:
        inspector = inspect(eng)
        return inspector.get_table_names()
    except exc.SQLAlchemyError as e:
        return f"Error getting schema: {e}"

def run_explain(sql_query: str, eng):
    """Runs EXPLAIN on a SQL query for the configured dialect."""
    if not eng: return "Error: Database engine not available."

    dialect = CONFIG["sql_dialect"]
    explain_query = ""
    if dialect == "postgresql":
        explain_query = f"EXPLAIN (VERBOSE, FORMAT JSON) {sql_query}"
    elif dialect == "mysql":
        explain_query = f"EXPLAIN FORMAT=JSON {sql_query}"
    else:
        # Fallback for other dialects like SQLite
        explain_query = f"EXPLAIN QUERY PLAN {sql_query}"

    try:
        with eng.connect() as connection:
            result = connection.execute(text(explain_query))
            return [row for row in result]
    except exc.SQLAlchemyError as e:
        return f"Error running EXPLAIN: {e}"

# --- 4. GRAPH STATE ---

class GraphState(TypedDict):
    sql_query: str
    db_schema: List[str]
    error_message: str
    is_final_sql: bool

# --- 5. PROMPTS ---

METADATA_CHECK_PROMPT = """
You are a database expert for the **{dialect}** SQL dialect.
Given a SQL query and the database schema (a list of table names),
check if the tables mentioned in the query exist in the schema.

Database Schema:
{schema}

SQL Query:
{query}

Respond with a JSON object containing two keys:
"is_valid": boolean (true if all tables exist, false otherwise)
"reason": string (explain why it's invalid, or "OK" if valid)
"""

BUSINESS_LOGIC_CHECK_PROMPT = """
You are a data privacy officer for a system using **{dialect}** SQL.
You must check if a SQL query violates any business logic rules.
The primary rule is: **Do not directly query Personally Identifiable Information (PII). A known PII column is 'email' in any 'users' table.**

SQL Query:
{query}

Does this query violate the business rule? Respond with a JSON object containing two keys:
"is_valid": boolean (true if it does NOT violate the rule, false if it does)
"reason": string (explain the violation, or "OK" if valid)
"""

# --- 6. GRAPH NODES ---

def check_syntax_node(state: GraphState):
    """1. Checks for basic SQL syntax."""
    print("--- 1. CHECKING SYNTAX ---")
    query = state['sql_query']
    if not re.match(r"\s*SELECT\s+.*\s+FROM\s+.*", query, re.IGNORECASE):
        state['error_message'] = "Syntax Error: Query must be a 'SELECT ... FROM ...' statement."
    else:
        state['error_message'] = ""
    return state

def check_metadata_node(state: GraphState):
    """2. Checks if tables and columns in the SQL exist in the DB schema."""
    print("--- 2. CHECKING METADATA ---")

    # Skip check if API key is a placeholder
    provider = CONFIG["llm_provider"]
    api_key = CONFIG["llm_configs"][provider].get("api_key", "")
    if api_key == "YOUR_OPENAI_API_KEY" or api_key == "sk-xxxx":
        print(f"Skipping metadata check: API key for {provider} is a placeholder.")
        state['error_message'] = ""
        return state

    query = state['sql_query']
    schema = get_schema(engine)
    state['db_schema'] = schema if isinstance(schema, list) else []

    if isinstance(schema, str) and "Error" in schema:
        state['error_message'] = f"Metadata Error: Could not retrieve schema. {schema}"
        return state

    prompt = METADATA_CHECK_PROMPT.format(
        dialect=CONFIG['sql_dialect'],
        schema=schema,
        query=query
    )
    response = llm.invoke(prompt)
    result = json.loads(response.content)

    if not result['is_valid']:
        state['error_message'] = f"Metadata Error: {result['reason']}"
    else:
        state['error_message'] = ""
    return state

def check_business_logic_node(state: GraphState):
    """3. Checks the SQL against business rules (e.g., PII)."""
    print("--- 3. CHECKING BUSINESS LOGIC ---")

    # Skip check if API key is a placeholder
    provider = CONFIG["llm_provider"]
    api_key = CONFIG["llm_configs"][provider].get("api_key", "")
    if api_key == "YOUR_OPENAI_API_KEY" or api_key == "sk-xxxx":
        print(f"Skipping business logic check: API key for {provider} is a placeholder.")
        state['error_message'] = ""
        return state

    query = state['sql_query']

    prompt = BUSINESS_LOGIC_CHECK_PROMPT.format(
        dialect=CONFIG['sql_dialect'],
        query=query
    )
    response = llm.invoke(prompt)
    result = json.loads(response.content)

    if not result['is_valid']:
        state['error_message'] = f"Business Logic Error: {result['reason']}"
    else:
        state['error_message'] = ""
    return state

def check_performance_node(state: GraphState):
    """4. Checks the query performance using EXPLAIN."""
    print("--- 4. CHECKING PERFORMANCE ---")
    query = state['sql_query']
    plan_result = run_explain(query, engine)

    if isinstance(plan_result, str) and "Error" in plan_result:
        state['error_message'] = f"Performance Check Error: {plan_result}"
        return state

    try:
        # This is a very basic check. Real-world checks would be more complex.
        # It assumes the JSON output from EXPLAIN is in the first column of the first row for pg/mysql
        # For SQLite, it checks the detail column of any row.
        plan_str = ""
        if CONFIG["sql_dialect"] in ["postgresql", "mysql"]:
            plan_json = json.loads(plan_result[0][0])
            plan_str = json.dumps(plan_json)
        else: # SQLite
            plan_str = " ".join(str(row) for row in plan_result)

        # A simplistic check for full table scans.
        # For PostgreSQL, "Seq Scan". For MySQL, "type": "ALL". For SQLite, "SCAN".
        if "Seq Scan" in plan_str or '"type": "ALL"' in plan_str or "SCAN" in plan_str:
             state['error_message'] = "Performance Warning: Query may involve a full table scan."
        else:
            state['error_message'] = ""
    except (json.JSONDecodeError, IndexError) as e:
        state['error_message'] = f"Performance Check Error: Could not parse EXPLAIN plan. {e}. Plan was: {plan_result}"

    return state

def success_node(state: GraphState):
    """Final node if all checks pass."""
    print("--- SUCCESS ---")
    state['is_final_sql'] = True
    return state

def error_node(state: GraphState):
    """Final node if any check fails."""
    print("--- ERROR ---")
    state['is_final_sql'] = False
    return state

# --- 7. GRAPH ASSEMBLY ---

def should_continue(state: GraphState):
    """Determines the next step based on whether an error was found."""
    if state.get('error_message'):
        return "error"
    return "continue"

workflow = StateGraph(GraphState)
workflow.add_node("check_syntax", check_syntax_node)
workflow.add_node("check_metadata", check_metadata_node)
workflow.add_node("check_business_logic", check_business_logic_node)
workflow.add_node("check_performance", check_performance_node)
workflow.add_node("success", success_node)
workflow.add_node("error", error_node)

workflow.set_entry_point("check_syntax")
workflow.add_conditional_edges("check_syntax", should_continue, {"continue": "check_metadata", "error": "error"})
workflow.add_conditional_edges("check_metadata", should_continue, {"continue": "check_business_logic", "error": "error"})
workflow.add_conditional_edges("check_business_logic", should_continue, {"continue": "check_performance", "error": "error"})
workflow.add_conditional_edges("check_performance", should_continue, {"continue": "success", "error": "error"})
workflow.add_edge("success", END)
workflow.add_edge("error", END)

app = workflow.compile()

# --- 8. MAIN EXECUTION (EXAMPLE) ---

def setup_test_sqlite_db(eng):
    """A helper to set up a temporary SQLite DB for testing purposes."""
    if eng.driver != "pysqlite":
        warnings.warn(f"Skipping test database setup because the configured database driver '{eng.driver}' is not 'pysqlite'.")
        return

    with eng.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS users"))
        conn.execute(text("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                username VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL,
                signup_date DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                order_date DATE,
                amount DECIMAL(10, 2),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )"""))
        conn.execute(text("INSERT INTO users (user_id, username, email, signup_date) VALUES (1, 'jules', 'jules@example.com', '2023-01-15')"))
        conn.execute(text("INSERT INTO orders (order_id, user_id, order_date, amount) VALUES (101, 1, '2023-03-10', 150.75)"))
        conn.commit()
    print("Test SQLite database set up successfully.")


if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    # To use a different configuration, modify the CONFIG dictionary at the top of the script.
    # For example, to use DeepSeek with a PostgreSQL database:
    #
    # CONFIG["llm_provider"] = "deepseek"
    # CONFIG["sql_dialect"] = "postgresql"
    # CONFIG["database_url"] = "postgresql://user:pass@localhost:5432/mydatabase"
    #
    # Make sure to replace the placeholder API keys in CONFIG["llm_configs"] with your actual keys.

    print("--- SQL Agent Initializing ---")
    print(f"  LLM Provider: {CONFIG['llm_provider']}")
    print(f"  SQL Dialect:  {CONFIG['sql_dialect']}")
    print(f"  Database URL: {CONFIG['database_url']}")
    print("------------------------------")

    # This check is important because the script requires a valid API key for most providers
    provider_config = CONFIG["llm_configs"][CONFIG["llm_provider"]]
    if "YOUR_API_KEY" in provider_config.get("api_key", "") or "sk-xxxx" in provider_config.get("api_key", ""):
        warnings.warn(f"API key for {CONFIG['llm_provider']} is a placeholder. LLM-dependent checks may fail.")

    if not engine:
        print("Could not initialize database engine. Exiting.")
    else:
        # For demonstration, if we are using SQLite, set up a test DB
        setup_test_sqlite_db(engine)

        # Example query to test
        sql_to_check = "SELECT user_id, username FROM users WHERE user_id = 1"

        print(f"\n=============================================")
        print(f"Testing Query: \"{sql_to_check}\"")
        print("=============================================")

        try:
            initial_state = {"sql_query": sql_to_check}
            final_state = app.invoke(initial_state)

            if final_state.get('is_final_sql'):
                print(f"\nResult: PASSED ✅")
                print(f"Final SQL: {final_state['sql_query']}")
            else:
                print(f"\nResult: FAILED ❌")
                print(f"Error: {final_state['error_message']}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during agent execution: {e}")
