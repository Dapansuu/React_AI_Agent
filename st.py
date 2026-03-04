import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from main import (
    chatbot,
    generate_chat_title,
    delete_conversation,
    conn,
)

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="ReAct Agent", layout="wide")

st.title("🧠 ReAct Agent")
st.markdown("Chat + Web Search + News + Currency Converter")

# --------------------------
# Initialize Session
# --------------------------
def initialize_session():

    # Always load conversations fresh
    rows = conn.execute(
        "SELECT thread_id, title FROM conversations"
    ).fetchall()

    st.session_state.conversations = {
        row[0]: {"title": row[1]}
        for row in rows
    }

    # Ensure current_chat exists and is valid
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
                "INSERT INTO conversations (thread_id, title) VALUES (?, ?)",
                (new_chat_id, "New Chat"),
            )
            conn.commit()

            st.session_state.current_chat = new_chat_id


initialize_session()

current_chat_id = st.session_state.current_chat
config = {"configurable": {"thread_id": current_chat_id}}

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("💬 Conversations")

# ➕ New Chat
if st.sidebar.button("➕ New Chat"):

    new_chat_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO conversations (thread_id, title) VALUES (?, ?)",
        (new_chat_id, "New Chat"),
    )
    conn.commit()

    st.session_state.current_chat = new_chat_id
    st.rerun()

# List conversations
for chat_id, chat_data in list(st.session_state.conversations.items()):

    col1, col2 = st.sidebar.columns([4, 1])

    if col1.button(chat_data["title"], key=f"select_{chat_id}"):
        st.session_state.current_chat = chat_id
        st.rerun()

    if col2.button("🗑", key=f"delete_{chat_id}"):

        delete_conversation(chat_id)

        st.session_state.pop("conversations", None)
        st.session_state.pop("current_chat", None)

        st.rerun()

# --------------------------
# Render Chat History
# --------------------------
state = chatbot.get_state(config)

messages = []
if state and "messages" in state.values:
    messages = state.values["messages"]

for message in messages:

    # Render only valid user messages
    if isinstance(message, HumanMessage) and message.content:
        with st.chat_message("user"):
            st.markdown(message.content)

    # Render only valid assistant messages (skip tool/empty)
    elif (
        isinstance(message, AIMessage)
        and message.content
        and message.content.strip() != ""
        and not getattr(message, "tool_calls", None)
    ):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --------------------------
# Chat Input
# --------------------------
user_input = st.chat_input("Ask something...")

if user_input:

    # Immediately show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant reply
    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):

            if (
                isinstance(message_chunk, AIMessage)
                and message_chunk.content
                and not getattr(message_chunk, "tool_calls", None)
            ):
                full_response += message_chunk.content
                placeholder.markdown(full_response)

    # Generate title if first interaction
    current_title = st.session_state.conversations.get(
        current_chat_id, {}
    ).get("title")

    if current_title == "New Chat":
        title = generate_chat_title(user_input)

        conn.execute(
            "UPDATE conversations SET title = ? WHERE thread_id = ?",
            (title, current_chat_id),
        )
        conn.commit()

    # Force clean rerun so history renders from DB state
    st.rerun()