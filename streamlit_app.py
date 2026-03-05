import streamlit as st
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from main import (
    chatbot,
    generate_chat_title,
    delete_conversation,
    conn,
    authenticate_user,
    create_user
)

st.set_page_config(page_title="ReAct Agent", layout="wide")


# ---------------- LOGIN STATE ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ---------------- AUTH PAGE ----------------
if not st.session_state.authenticated:

    st.title("🔐 ReAct Agent Authentication")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # LOGIN
    with tab1:

        login_id = st.text_input("User ID")
        login_pass = st.text_input("Password", type="password")

        if st.button("Login"):

            if authenticate_user(login_id, login_pass):

                st.session_state.authenticated = True
                st.session_state.user = login_id

                st.rerun()

            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:

        new_id = st.text_input("Create User ID")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account"):

            if len(new_id) < 3 or len(new_pass) < 4:
                st.warning("User ID or password too short")

            else:

                success = create_user(new_id, new_pass)

                if success:
                    st.success("Account created. You can now login.")

                else:
                    st.error("User already exists")

    st.stop()


# ---------------- APP ----------------
st.title("🧠 ReAct Agent")
st.markdown("Chat + Web Search + News + Currency Converter")

# ---------------- SIDEBAR ----------------
st.sidebar.title("💬 Conversations")

st.sidebar.write(f"👤 {st.session_state.user}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()


# ---------------- SESSION INIT ----------------
def initialize_session():

    rows = conn.execute(
        "SELECT thread_id, title FROM conversations"
    ).fetchall()

    st.session_state.conversations = {
        row[0]: {"title": row[1]}
        for row in rows
    }

    if (
        "current_chat" not in st.session_state
        or st.session_state.current_chat
        not in st.session_state.conversations
    ):

        if st.session_state.conversations:

            st.session_state.current_chat = list(
                st.session_state.conversations.keys()
            )[0]

        else:

            new_chat_id = str(uuid.uuid4())

            conn.execute(
                "INSERT INTO conversations VALUES (?, ?)",
                (new_chat_id, "New Chat")
            )

            conn.commit()

            st.session_state.current_chat = new_chat_id


initialize_session()

current_chat_id = st.session_state.current_chat

config = {"configurable": {"thread_id": current_chat_id}}


# ---------------- NEW CHAT ----------------
if st.sidebar.button("➕ New Chat"):

    new_chat_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO conversations VALUES (?, ?)",
        (new_chat_id, "New Chat")
    )

    conn.commit()

    st.session_state.current_chat = new_chat_id
    st.rerun()


# ---------------- CONVERSATION LIST ----------------
for chat_id, chat_data in list(st.session_state.conversations.items()):

    col1, col2 = st.sidebar.columns([4,1])

    if col1.button(chat_data["title"], key=f"select_{chat_id}"):

        st.session_state.current_chat = chat_id
        st.rerun()

    if col2.button("🗑", key=f"delete_{chat_id}"):

        delete_conversation(chat_id)

        st.session_state.pop("conversations", None)
        st.session_state.pop("current_chat", None)

        st.rerun()


# ---------------- HISTORY ----------------
state = chatbot.get_state(config)

messages = []

if state and "messages" in state.values:
    messages = state.values["messages"]

for message in messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage) and message.content:

        if not getattr(message, "tool_calls", None):

            with st.chat_message("assistant"):
                st.markdown(message.content)


# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for chunk, meta in chatbot.stream(
            {"messages":[HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):

            if isinstance(chunk, AIMessage) and chunk.content:

                if not getattr(chunk,"tool_calls",None):

                    full_response += chunk.content
                    placeholder.markdown(full_response)

    current_title = st.session_state.conversations.get(
        current_chat_id, {}
    ).get("title")

    if current_title == "New Chat":

        title = generate_chat_title(user_input)

        conn.execute(
            "UPDATE conversations SET title=? WHERE thread_id=?",
            (title,current_chat_id)
        )

        conn.commit()

    st.rerun()