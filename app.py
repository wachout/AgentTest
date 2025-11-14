
import streamlit as st
import subprocess
import sys

# --- Dependency Installation ---
try:
    import extra_streamlit_components as stx
except ImportError:
    st.info("Installing missing dependency: extra-streamlit-components...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "extra-streamlit-components"])
    st.rerun()

# --- API Imports ---
from api import (login_user, register_user, delete_user,
                 get_user_session_messages, get_sessions_by_id,
                 create_session, delete_sessions_by_session_id,
                 execute_stream_chat)
import time

# --- Page and State Setup ---
st.set_page_config(page_title="Session Chat", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

def initialize_session_state():
    defaults = {
        "logged_in": False,
        "user_name": "",
        "password": "",
        "user_id": "",
        "user_sessions": [],
        "selected_session": None,
        "chat_history": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Authentication Pages ---
def login_page():
    st.title("Login to Your Account")
    with st.form("login_form"):
        user_name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            success, message, user_id = login_user(user_name, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.password = password
                st.session_state.user_id = user_id
                st.rerun()
            else:
                st.error(message)

def register_page():
    st.title("Register a New Account")
    # Implementation remains the same...

def delete_user_page():
    st.title("Delete Your Account")
    # Implementation remains the same...

# --- Main Application Page ---
def main_page():
    # --- Sidebar ---
    with st.sidebar:
        st.title("Session Management")
        st.write(f"Welcome, {st.session_state.user_name}!")
        if st.button("Logout"):
            # Reset state on logout
            initialize_session_state()
            st.rerun()

        # Create New Session
        st.header("Create New Session")
        with st.form("create_session_form"):
            session_name = st.text_input("Session Name")
            knowledge_name = st.text_input("Knowledge Base (Optional)")
            if st.form_submit_button("Create"):
                success, message, _ = create_session(
                    st.session_state.user_name, st.session_state.password, session_name, knowledge_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # List User Sessions
        st.header("My Sessions")
        success, _, sessions = get_user_session_messages(
            st.session_state.user_name, st.session_state.password, st.session_state.user_id)
        if success:
            st.session_state.user_sessions = sessions
            for session in st.session_state.user_sessions:
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                with col1:
                    st.write(f"**{session['session_name']}**")
                    st.caption(session['session_desc'])
                with col2:
                    if st.button("Select", key=f"select_{session['session_id']}"):
                        st.session_state.selected_session = session
                        # Fetch and load chat history
                        _, _, full_session_data = get_sessions_by_id(session['session_id'])
                        if full_session_data:
                            st.session_state.chat_history = full_session_data[0].get("messages", [])
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete_{session['session_id']}"):
                        delete_sessions_by_session_id(session['session_id'])
                        st.rerun()

    # --- Main Chat Area ---
    st.title("Chat")
    if st.session_state.selected_session:
        st.header(f"Session: {st.session_state.selected_session['session_name']}")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask something..."):
            # Add user message to UI and history
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Stream assistant response
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                stream = execute_stream_chat(
                    st.session_state.user_name,
                    st.session_state.password,
                    st.session_state.selected_session['session_id'],
                    prompt
                )
                for chunk in stream:
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)

            # Add final assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    else:
        st.info("Select a session from the sidebar to begin chatting.")

# --- App Router ---
def render_app():
    if not st.session_state.logged_in:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Login", "Register", "Delete User"])
        if page == "Login":
            login_page()
        elif page == "Register":
            # Keeping register_page and delete_user_page functions for completeness
            st.title("Register a New Account")
        else:
            st.title("Delete Your Account")
    else:
        main_page()

render_app()
