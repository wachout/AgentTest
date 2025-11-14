
import streamlit as st
import subprocess
import sys

try:
    import extra_streamlit_components as stx
except ImportError:
    st.info("Installing missing dependency: extra-streamlit-components")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "extra-streamlit-components"])
    st.rerun()

from api import (login_user, register_user, delete_user, create_knowledge_base,
                 delete_knowledge_base, get_user_knowledge_bases, add_file,
                 execute_stream_chat, update_knowledge_base, delete_file)
import time

st.set_page_config(page_title="Knowledge Base", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.password = ""
        st.session_state.user_id = ""
        st.session_state.session_id = ""
        st.session_state.knowledge_bases = []
        st.session_state.selected_kb = None
        st.session_state.sessions = {}
        st.session_state.current_session_index = -1

initialize_session_state()

def login_page():
    st.title("Login to Your Account")
    with st.form("login_form"):
        user_name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            success, message, user_id, session_id = login_user(user_name, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.password = password
                st.session_state.user_id = user_id
                st.session_state.session_id = session_id
                st.rerun()
            else:
                st.error(message)

def register_page():
    st.title("Register a New Account")
    with st.form("register_form"):
        user_name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Register"):
            success, message, _ = register_user(user_name, password, confirm_password)
            if success:
                st.success(message)
                st.info("You can now log in with your new account.")
            else:
                st.error(message)

def delete_user_page():
    st.title("Delete Your Account")
    with st.form("delete_user_form"):
        user_name = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Delete Account"):
            success, message = delete_user(user_name, password)
            if success:
                st.success(message)
                st.info("Your account has been deleted.")
                time.sleep(2)
                st.session_state.logged_in = False
                st.rerun()
            else:
                st.error(message)

def main_page():
    with st.sidebar:
        st.title("Knowledge Base Management")
        st.write(f"Welcome, {st.session_state.user_name}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        st.header("Create Knowledge Base")
        with st.form("create_kb_form"):
            kb_name = st.text_input("Name")
            kb_desc = st.text_area("Description")
            start_time = st.text_input("Valid Start Time (YYYY-MM-DD HH:MM:SS)")
            end_time = st.text_input("Valid End Time (YYYY-MM-DD HH:MM:SS)")
            if st.form_submit_button("Create"):
                success, message, _ = create_knowledge_base(st.session_state.user_name, st.session_state.password, kb_name, kb_desc, start_time, end_time)
                st.success(message) if success else st.error(message)

        st.header("Your Knowledge Bases")
        success, _, kbs = get_user_knowledge_bases(st.session_state.user_name, st.session_state.password)
        if success:
            st.session_state.knowledge_bases = kbs
            for kb in kbs:
                with st.expander(kb['name']):
                    st.write(f"**Description:** {kb['description']}")
                    if st.button(f"Select {kb['name']}", key=f"select_{kb['knowledge_id']}"):
                        st.session_state.selected_kb = kb
                    if st.button(f"Delete {kb['name']}", key=f"delete_{kb['knowledge_id']}"):
                        success, message = delete_knowledge_base(st.session_state.user_name, st.session_state.password, kb['name'])
                        st.success(message) if success else st.error(message)

                    with st.form(f"update_kb_form_{kb['knowledge_id']}"):
                        st.subheader("Update Knowledge Base")
                        new_name = st.text_input("New Name", value=kb['name'])
                        new_desc = st.text_area("New Description", value=kb['description'])
                        new_start_time = st.text_input("New Valid Start Time", value=kb['valid_start_time'])
                        new_end_time = st.text_input("New Valid End Time", value=kb['valid_end_time'])
                        if st.form_submit_button("Update"):
                            success, message = update_knowledge_base(st.session_state.user_name, st.session_state.password, kb['name'], new_name, new_desc, new_start_time, new_end_time)
                            st.success(message) if success else st.error(message)

        st.header("Upload File")
        if uploaded_file := st.file_uploader("Choose a file"):
            if st.session_state.selected_kb:
                file_path = f"/tmp/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                success, message = add_file(uploaded_file.name, file_path, st.session_state.user_name, st.session_state.password)
                st.success(message) if success else st.error(message)
            else:
                st.warning("Please select a knowledge base first.")

        if st.session_state.selected_kb:
            st.header("Files in Knowledge Base")
            # In a real app, you'd fetch this list from the backend
            files = [{"file_id": "file1", "name": "example.txt"}]
            for file in files:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(file["name"])
                with col2:
                    if st.button(f"Delete", key=f"delete_file_{file['file_id']}"):
                        success, message = delete_file(st.session_state.user_name, st.session_state.password, file['file_id'])
                        st.success(message) if success else st.error(message)

    st.title("Chat")
    if st.session_state.selected_kb:
        st.header(f"Chatting with: {st.session_state.selected_kb['name']}")
        kb_id = st.session_state.selected_kb['knowledge_id']
        if kb_id not in st.session_state.sessions:
            st.session_state.sessions[kb_id] = [[]]

        if st.button("New Session"):
            st.session_state.sessions[kb_id].append([])
            st.session_state.current_session_index = len(st.session_state.sessions[kb_id]) - 1

        session_tabs = [f"Session {i+1}" for i, _ in enumerate(st.session_state.sessions[kb_id])]
        chosen_id = stx.tab_bar(data=[stx.TabBarItemData(id=i, title=tab, description="") for i, tab in enumerate(session_tabs)])

        if chosen_id is not None:
            st.session_state.current_session_index = int(chosen_id)
            chat_history = st.session_state.sessions[kb_id][st.session_state.current_session_index]
            for author, message in chat_history:
                with st.chat_message(author):
                    st.write(message)

            if prompt := st.chat_input("What is up?"):
                chat_history.append(("user", prompt))
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    response_container = st.empty()
                    full_response = ""
                    for chunk in execute_stream_chat(st.session_state.user_name, st.session_state.password, st.session_state.selected_kb['name'], st.session_state.session_id, prompt):
                        full_response += chunk
                        response_container.markdown(full_response)
                    chat_history.append(("assistant", full_response))
    else:
        st.info("Select a knowledge base from the sidebar to start chatting.")

def render_app():
    if not st.session_state.logged_in:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Login", "Register", "Delete User"])
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
        else:
            delete_user_page()
    else:
        main_page()

render_app()
