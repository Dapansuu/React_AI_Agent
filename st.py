import streamlit as st
from langchain_core.messages import HumanMessage
from main import chatbot

config = {"configurable": {"thread_id": None}}

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="AI Agent", layout="centered")

st.title("ReAct Agent")
st.markdown("Chat + Web Search + News + Currency Converter")

# --------------------------
# Session State
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------
# Chat Display
# --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------
# Chat Input
# --------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke LangGraph agent
    result = chatbot.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config = config)

    final_response = result["messages"][-1].content

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_response}
    )
