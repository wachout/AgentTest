
import time

def login_user(user_name, password):
    if user_name == "testuser" and password == "testpassword":
        return True, "Login successful", "123", "abc"
    return False, "Invalid credentials", None, None

def register_user(user_name, password, confirm_password):
    return True, "User registered successfully", "123"

def delete_user(user_name, password):
    return True, "User deleted successfully"

def create_knowledge_base(user_name, password, name, description, start_time, end_time):
    return True, f"Knowledge base '{name}' created successfully", name

def delete_knowledge_base(user_name, password, knowledge_name):
    return True, f"Knowledge base '{knowledge_name}' deleted successfully"

def get_user_knowledge_bases(user_name, password):
    return True, "", [
        {
            "knowledge_id": "kb1",
            "name": "Test KB",
            "description": "A test knowledge base.",
            "valid_start_time": "2023-01-01 00:00:00",
            "valid_end_time": "2025-12-31 23:59:59",
        }
    ]

def get_knowledge_base_by_name(knowledge_name):
    return True, "", {
        "knowledge_id": "kb1",
        "name": "Test KB",
        "description": "A test knowledge base.",
        "valid_start_time": "2023-01-01 00:00:00",
        "valid_end_time": "2025-12-31 23:59:59",
    }

def add_file(file_name, file_path, user_name, password):
    return True, f"File '{file_name}' added successfully"

def execute_stream_chat(user_name, password, knowledge_name, session_id, query):
    response = "This is a simulated streaming response. " * 5
    for char in response:
        yield char
        time.sleep(0.05)

def update_knowledge_base(user_name, password, knowledge_name, new_name, new_description, new_start_time, new_end_time):
    return True, f"Knowledge base '{knowledge_name}' updated successfully"

def delete_file(user_name, password, file_id):
    return True, f"File with ID '{file_id}' deleted successfully"
