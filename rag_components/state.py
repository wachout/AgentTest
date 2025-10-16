from typing import List, Dict, Any, Optional

class State:
    """
    Manages the state of the RAG agent's conversation flow.
    """
    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []
        self.retrieved_docs: List[Any] = []
        self.document_relevance: bool = False
        self.generation: Optional[str] = None
        self.refinement_attempts: int = 0

    def add_message(self, role: str, content: str):
        """Adds a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})

    def get_last_user_message(self) -> Optional[str]:
        """Retrieves the last user message from the history."""
        for message in reversed(self.chat_history):
            if message["role"] == "user":
                return message["content"]
        return None

    def __repr__(self) -> str:
        return (
            f"State(chat_history={self.chat_history}, "
            f"retrieved_docs={len(self.retrieved_docs)}, "
            f"document_relevance={self.document_relevance}, "
            f"refinement_attempts={self.refinement_attempts})"
        )