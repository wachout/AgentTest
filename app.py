import streamlit as st
from api import api
import extra_streamlit_components as stx

st.set_page_config(layout="wide", page_title="大模型聊天界面")


def login_page():
    st.title("登录")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
        if submitted:
            response = api.user_login(username, password)
            if response["success"]:
                st.session_state.logged_in = True
                st.session_state.user_id = response["user_id"]
                st.session_state.username = username
                st.success(response["message"])
                st.rerun()
            else:
                st.error(response["message"])


def register_page():
    st.title("注册")
    with st.form("register_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        submitted = st.form_submit_button("注册")
        if submitted:
            response = api.register(username, password, confirm_password)
            if response["success"]:
                st.success(response["message"])
            else:
                st.error(response["message"])


def delete_user_page():
    st.title("删除用户")
    with st.form("delete_user_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("删除")
        if submitted:
            response = api.delete_user(username, password)
            if response["success"]:
                st.success(response["message"])
            else:
                st.error(response["message"])


def main_page():
    # Load CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.sidebar.title(f"欢迎, {st.session_state.username}")
    if st.sidebar.button("登出"):
        api.user_logout(st.session_state.get("session_id"))
        st.session_state.logged_in = False
        st.session_state.clear()
        st.rerun()

    # Initialize session state variables
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for knowledge base and session management
    with st.sidebar:
        st.header("知识库")
        st.selectbox("选择知识库", ["默认知识库", "知识库 A", "知识库 B"])

        st.header("会话列表")

        # Create new session
        new_session_name = st.text_input("新会话名称")
        if st.button("创建新会话"):
            if new_session_name:
                response = api.create_session(
                    st.session_state.username, "", new_session_name
                )
                if response["success"]:
                    st.success("新会话已创建！")
                    st.rerun()
                else:
                    st.error(response["message"])
            else:
                st.warning("请输入会话名称。")

        # Fetch and display session list
        sessions_response = api.get_session_messages(st.session_state.username, "")
        if sessions_response["success"]:
            st.session_state.sessions = sessions_response["messages"]
            session_options = {
                s["session_id"]: s["session_name"] for s in st.session_state.sessions
            }

            if st.session_state.sessions:
                selected_session_id = st.radio(
                    "选择会话",
                    options=session_options.keys(),
                    format_func=lambda session_id: session_options[session_id],
                )

                if selected_session_id != st.session_state.selected_session_id:
                    st.session_state.selected_session_id = selected_session_id
                    history_response = api.get_sessions_by_id(selected_session_id)
                    if history_response["success"] and history_response["messages"]:
                        st.session_state.messages = history_response["messages"][0].get(
                            "messages", []
                        )
                    else:
                        st.session_state.messages = []
                        st.error("无法加载会话历史。")

                # Delete session
                if st.button("删除当前会话"):
                    delete_response = api.delete_sessions_by_session_id(
                        st.session_state.selected_session_id
                    )
                    if delete_response["success"]:
                        st.success("会话已删除！")
                        st.session_state.selected_session_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error(delete_response["message"])
        else:
            st.error("无法加载会话列表。")

    # Main chat area
    selected_session_name = "未选择会话"
    if st.session_state.selected_session_id:
        for s in st.session_state.sessions:
            if s["session_id"] == st.session_state.selected_session_id:
                selected_session_name = s["session_name"]
                break
    st.header(f"聊天窗口 - {selected_session_name}")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if st.session_state.selected_session_id:
        prompt = st.chat_input("请输入您的问题...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in api.chat(
                    user_name=st.session_state.username,
                    password="",
                    session_id=st.session_state.selected_session_id,
                    query=prompt,
                ):
                    if response["success"]:
                        content = response["data"]["choices"][0]["delta"].get(
                            "content", ""
                        )
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        st.chat_input("请先选择或创建一个会话", disabled=True)

    # File upload
    with st.expander("上传文件"):
        uploaded_file = st.file_uploader("选择一个文件", type=["txt", "pdf", "md"])
        if uploaded_file is not None:
            st.success(f"文件 '{uploaded_file.name}' 上传成功！")
            save_option = st.radio(
                "是否保存到知识库？",
                ("不保存", "私人可见", "共享可见"),
                key=f"save_option_{uploaded_file.name}",  # Use a unique key
            )
            if st.button("确认", key=f"confirm_button_{uploaded_file.name}"):
                if save_option != "不保存":
                    # Here you would typically call an API to save the file
                    st.success(f"文件已保存为 '{save_option}'。")
                else:
                    st.info("文件未保存。")


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        chosen_id = stx.tab_bar(
            data=[
                stx.TabBarItemData(id=1, title="登录", description=""),
                stx.TabBarItemData(id=2, title="注册", description=""),
                stx.TabBarItemData(id=3, title="删除用户", description=""),
            ],
            default=1,
        )

        if chosen_id == "1":
            login_page()
        elif chosen_id == "2":
            register_page()
        elif chosen_id == "3":
            delete_user_page()
    else:
        main_page()


if __name__ == "__main__":
    main()
