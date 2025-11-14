import requests
import json
from config import BASE_URL


class API:
    def _post(self, endpoint, data):
        try:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "message": str(e)}

    def user_login(self, user_name, password):
        return self._post("/user_login", {"user_name": user_name, "password": password})

    def user_logout(self, session_id):
        return self._post("/user_logout", {"session_id": session_id})

    def register(self, user_name, password, confirm_password):
        return self._post(
            "/register",
            {
                "user_name": user_name,
                "password": password,
                "confirm_password": confirm_password,
            },
        )

    def delete_user(self, user_name, password):
        return self._post(
            "/delete_user", {"user_name": user_name, "password": password}
        )

    def create_session(
        self,
        user_name,
        password,
        session_name,
        konwledge_name=None,
        knowledge_id=None,
        user_id=None,
    ):
        data = {
            "user_name": user_name,
            "password": password,
            "session_name": session_name,
        }
        if konwledge_name:
            data["konwledge_name"] = konwledge_name
        if knowledge_id:
            data["knowledge_id"] = knowledge_id
        if user_id:
            data["user_id"] = user_id
        return self._post("/create_session", data)

    def get_session_messages(self, user_name, password, user_id=None):
        data = {"user_name": user_name, "password": password}
        if user_id:
            data["user_id"] = user_id
        return self._post("/get_session_messages", data)

    def get_sessions_by_id(self, session_id):
        return self._post("/get_sessions_by_id", {"session_id": session_id})

    def delete_sessions_by_session_id(self, session_id):
        return self._post("/delete_sessions_by_session_id", {"session_id": session_id})

    def chat(
        self,
        user_name,
        password,
        session_id,
        query,
        knowledge_name=None,
        knowledge_id=None,
        stream_chat=True,
        stream_chat_type=None,
    ):
        data = {
            "user_name": user_name,
            "password": password,
            "session_id": session_id,
            "query": query,
            "stream_chat": stream_chat,
            "stream_chat_type": stream_chat_type,
        }
        if knowledge_name:
            data["knowledge_name"] = knowledge_name
        if knowledge_id:
            data["knowledge_id"] = knowledge_id

        try:
            response = requests.post(f"{BASE_URL}/chat", json=data, stream=True)
            response.raise_for_status()
            for chunk in response.iter_lines():
                if chunk:
                    yield json.loads(chunk)
        except requests.exceptions.RequestException as e:
            yield {"success": False, "message": str(e)}
        except json.JSONDecodeError as e:
            yield {"success": False, "message": f"Error decoding JSON: {e}"}


api = API()
