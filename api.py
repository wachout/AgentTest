
import time
import uuid

# --- Mock Data Store ---
MOCK_SESSIONS = {
    "session-1": {
        "session_id": "session-1",
        "session_name": "Greeting",
        "session_desc": "Initial conversation",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么需要帮助的吗？"}
        ]
    },
    "session-2": {
        "session_id": "session-2",
        "session_name": "Follow-up",
        "session_desc": "A second conversation",
        "messages": []
    }
}

# --- User and Auth ---
def login_user(user_name, password):
    if user_name == "testuser" and password == "testpassword":
        return True, "Login successful", "user-123"
    return False, "Invalid credentials", None

def register_user(user_name, password, confirm_password):
    return True, "User registered successfully", "user-123"

def delete_user(user_name, password):
    return True, "User deleted successfully"

# --- Session Management ---
def get_user_session_messages(user_name, password, user_id=None):
    # Returns a list of session summaries
    session_list = [
        {"session_id": data["session_id"], "session_name": data["session_name"], "session_desc": data["session_desc"]}
        for _, data in MOCK_SESSIONS.items()
    ]
    return True, "Sessions retrieved successfully", session_list

def get_sessions_by_id(session_id):
    session_data = MOCK_SESSIONS.get(session_id)
    if session_data:
        # Returns the full session details including messages
        return True, "Session found", [session_data]
    return False, "Session not found", []

def create_session(user_name, password, session_name, konwledge_name=None, knowledge_id=None, user_id=None):
    new_id = f"session-{uuid.uuid4().hex[:6]}"
    MOCK_SESSIONS[new_id] = {
        "session_id": new_id,
        "session_name": session_name,
        "session_desc": f"About {konwledge_name}" if konwledge_name else "General conversation",
        "messages": []
    }
    return True, "Session created successfully", new_id

def delete_sessions_by_session_id(session_id):
    if session_id in MOCK_SESSIONS:
        del MOCK_SESSIONS[session_id]
        return True, "Session deleted successfully"
    return False, "Session not found"

# --- Knowledge Base (Legacy, kept for reference) ---
def create_knowledge_base(user_name, password, name, description, start_time, end_time):
    return True, f"Knowledge base '{name}' created successfully", name

def delete_knowledge_base(user_name, password, knowledge_name):
    return True, f"Knowledge base '{knowledge_name}' deleted successfully"

def get_user_knowledge_bases(user_name, password):
    return True, "", [{"knowledge_id": "kb1", "name": "Test KB", "description": "A test KB."}]

# --- File Management (Legacy, kept for reference) ---
def add_file(file_name, file_path, user_name, password):
    return True, f"File '{file_name}' added successfully"

def delete_file(user_name, password, file_id):
    return True, f"File with ID '{file_id}' deleted successfully"

# --- Chat Execution ---
def execute_stream_chat(user_name, password, session_id, query, knowledge_name=None):
    # Append the user's query to the mock session
    if session_id in MOCK_SESSIONS:
        MOCK_SESSIONS[session_id]["messages"].append({"role": "user", "content": query})

    response_text = "This is a simulated streaming response to your query. "
    # Add some dynamic content based on the query for realism
    if "hello" in query.lower():
        response_text = "Hello back to you! "
    elif "help" in query.lower():
        response_text = "How can I assist you today? "

    full_response = response_text * 3

    # Simulate streaming
    for char in full_response:
        yield char
        time.sleep(0.02)

    # Append the assistant's response to the mock session
    if session_id in MOCK_SESSIONS:
        MOCK_SESSIONS[session_id]["messages"].append({"role": "assistant", "content": full_response})
