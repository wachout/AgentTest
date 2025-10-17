from .state import AgentState

def retrieve_content(state: AgentState):
    """
    Retrieves content from the knowledge base.
    """
    # This is a mock implementation.
    knowledge_base = {
        "how to reset password": "To reset your password, go to the settings page and click on 'Reset Password'.",
        "how to update profile": "You can update your profile from the 'Profile' section in your account.",
        "billing issues": "For any billing issues, please contact our support team at support@example.com."
    }

    retrieved = []
    query = state["enhanced_query"]
    for key, value in knowledge_base.items():
        if any(word in query.lower() for word in key.split()):
            retrieved.append({"content": value})

    return {"retrieved_docs": retrieved}