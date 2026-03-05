import streamlit as st
import sqlite3
import hashlib
import json
import uuid
import os
import re
import requests
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
import operator

from dotenv import load_dotenv
load_dotenv()  

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from ddgs import DDGS
import feedparser
# ─────────────────────────────────────────────
# CONFIG  –  safe key reader (works before Streamlit fully initialises)
# ─────────────────────────────────────────────
def _get_secret(key: str) -> str:
    """Read a secret from env-vars first, then st.secrets (gracefully)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        val = st.secrets.get(key, "")
        return val or ""
    except Exception:
        return ""

def OPENROUTER_API_KEY():  return _get_secret("OPENROUTER_API_KEY")
def OPENWEATHER_API_KEY(): return _get_secret("OPENWEATHER_API_KEY")
def NEWS_API_KEY():        return _get_secret("NEWS_API_KEY")
# Stock / currency use free, no-key APIs (Yahoo Finance scrape + frankfurter.app)

DB_PATH = "chat_agent.db"

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TEXT NOT NULL
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            title      TEXT NOT NULL,
            created    TEXT NOT NULL,
            updated    TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created         TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )""")
    conn.commit()
    conn.close()

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        uid = str(uuid.uuid4())
        c.execute("INSERT INTO users VALUES (?,?,?,?)",
                    (uid, username, hash_password(password), datetime.now().isoformat()))
        conn.commit()
        return uid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?",
                (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def get_conversations(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,title,updated FROM conversations WHERE user_id=? ORDER BY updated DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def create_conversation(user_id, title="New Chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    c.execute("INSERT INTO conversations VALUES (?,?,?,?,?)", (cid, user_id, title, now, now))
    conn.commit()
    conn.close()
    return cid

def delete_conversation(conv_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()

def update_conversation_title(conv_id, title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title=?, updated=? WHERE id=?",
                (title[:60], datetime.now().isoformat(), conv_id))
    conn.commit()
    conn.close()

def save_message(conv_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    mid = str(uuid.uuid4())
    c.execute("INSERT INTO messages VALUES (?,?,?,?,?)",
                (mid, conv_id, role, content, datetime.now().isoformat()))
    c.execute("UPDATE conversations SET updated=? WHERE id=?",
                (datetime.now().isoformat(), conv_id))
    conn.commit()
    conn.close()

def load_messages(conv_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY created", (conv_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def search_tool(query: str): 
    with DDGS() as ddgs: 
        results = ddgs.text(query, max_results=5) 
        return list(results)
@tool
def web_search(query: str) -> str:
    """
    REQUIRED for any real-time or factual lookup.

    MUST be used for:
    - current events
    - factual verification
    - people in office
    - latest updates
    - statistics
    - general web lookups
    - "who is", "what is", "when did", etc.
    - anything that may have changed after training

    Never answer these from memory.
    Always call this tool when fresh or external information is required.
    """
    results = search_tool(query)
    formatted = [] 
    for r in results: 
        formatted.append( 
                    f"Title: {r.get('title')}\n" 
                    f"Snippet: {r.get('body')}\n" 
                    f"Link: {r.get('href')}\n" ) 
    return "\n\n".join(formatted)

@tool
def news(topic: str) -> str:
    """
    REQUIRED for all news-related queries.
    Always retrieves recent news (last 24 hours).
    """

    try:
        import urllib.parse
        from datetime import datetime
        import feedparser

        # Force last 24 hours
        query = f"{topic} when:1d"
        encoded_query = urllib.parse.quote(query)

        url = (
            f"https://news.google.com/rss/search?q={encoded_query}"
            f"&hl=en-IN&gl=IN&ceid=IN:en"
        )

        feed = feedparser.parse(url)

        if not feed.entries:
            return f"No news found in the last 24 hours for '{topic}'."

        articles = []

        for entry in feed.entries[:5]:
            published = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M")

            articles.append(
                f"""
            Title: {entry.title}
            Published: {published}
            Source: {entry.link}
            """.strip()
            )

        return "\n\n---\n\n".join(articles)

    except Exception as e:
        return f"News retrieval error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic info for a ticker symbol using Yahoo Finance (free, no API key)."""
    try:
        sym = symbol.upper().strip()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        meta = data["chart"]["result"][0]["meta"]
        price        = meta.get("regularMarketPrice", "N/A")
        prev_close   = meta.get("chartPreviousClose", "N/A")
        currency     = meta.get("currency", "USD")
        exchange     = meta.get("exchangeName", "N/A")
        name         = meta.get("shortName", sym)
        change       = round(float(price) - float(prev_close), 2) if price != "N/A" and prev_close != "N/A" else "N/A"
        pct          = round((change / float(prev_close)) * 100, 2) if change != "N/A" and prev_close else "N/A"
        return (f"📈 {name} ({sym})\n"
                f"Exchange : {exchange}\n"
                f"Price    : {price} {currency}\n"
                f"Prev Close: {prev_close} {currency}\n"
                f"Change   : {change} ({pct}%)")
    except Exception as e:
        return f"Stock lookup error for '{symbol}': {str(e)}"

@tool
def get_currency_exchange(base: str, target: str, amount: float = 1.0) -> str:
    """Convert between currencies using Frankfurter API (free, no API key needed)."""
    try:
        b = base.upper().strip()
        t = target.upper().strip()
        url = f"https://api.frankfurter.app/latest?from={b}&to={t}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "rates" not in data:
            return f"Could not fetch rate for {b} → {t}. Check currency codes."
        rate = data["rates"][t]
        converted = round(amount * rate, 4)
        date = data.get("date", "today")
        return (f"💱 Currency Exchange ({date})\n"
                f"{amount} {b} = {converted} {t}\n"
                f"Rate: 1 {b} = {rate} {t}")
    except Exception as e:
        return f"Currency exchange error: {str(e)}"

@tool
def weather(city: str) -> str:
    """
    REQUIRED for all weather-related queries.

    MUST be used when:
    - user asks about weather
    - temperature
    - forecast
    - rain
    - humidity
    - wind
    - climate conditions
    - "weather in [city]"
    - "is it raining in [city]"
    - "temperature in [city]"

    Never answer weather questions from memory.
    Always use this tool for live weather data.

    Input format:
    city name (e.g., "Delhi", "New York", "London")
    """

    try:
        # 1️⃣ Geocode city to get latitude & longitude
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1
        }

        geo_response = requests.get(geo_url, params=geo_params).json()

        if "results" not in geo_response:
            return "City not found."

        lat = geo_response["results"][0]["latitude"]
        lon = geo_response["results"][0]["longitude"]
        location_name = geo_response["results"][0]["name"]
        country = geo_response["results"][0]["country"]

        # 2️⃣ Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True
        }

        weather_response = requests.get(weather_url, params=weather_params).json()

        current = weather_response.get("current_weather")

        if not current:
            return "Weather data unavailable."

        temperature = current["temperature"]
        windspeed = current["windspeed"]
        weather_code = current["weathercode"]

        return (
            f"Weather in {location_name}, {country}:\n"
            f"Temperature: {temperature}°C\n"
            f"Wind Speed: {windspeed} km/h\n"
            f"Weather Code: {weather_code}"
        )

    except Exception as e:
        return "Error: " + str(e)

# ─────────────────────────────────────────────
# AGENT STATE & GRAPH
# ─────────────────────────────────────────────
TOOLS = [web_search, news, get_stock_price, get_currency_exchange, weather]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    validation_passed: bool
    validation_note: str

SYSTEM_PROMPT = """You are a highly capable AI assistant with access to real-time tools:
- web_search: search the internet for any information
- get_news: fetch latest news headlines on any topic
- get_stock_price: get live stock prices (e.g., AAPL, TSLA, GOOGL)
- get_currency_exchange: convert between any currencies
- get_weather: get current weather for any city

## STRICT TOOL-USE RULES - follow these without exception:

1. NEVER answer from memory for anything that can change over time. This includes:
    - Current officeholders (CM, PM, President, CEO, ministers, governors, etc.)
    - Prices, rates, scores, rankings, statistics
    - Recent events, news, policies, appointments
    - Any question containing words like: current, now, today, latest, recent, who is, still

2. ALWAYS call web_search FIRST before answering questions about:
    - People in any role or position ("CM of X", "PM of Y", "CEO of Z")
    - Any fact that may have changed since 2023
    - Current affairs, politics, sports results, business news

3. Your training data is outdated. Do NOT trust it for facts about the present.
    If you think you know the answer - search anyway to verify before responding.

4. After receiving tool results, synthesize them into a clear, helpful, well-formatted response.
    Be concise yet thorough. Use bullet points and emojis where appropriate.

EXAMPLE of correct behaviour:
User: "Who is the CM of Haryana?"
WRONG: Answer from memory (stale, unreliable)
CORRECT: Call web_search("current Chief Minister of Haryana 2024") then answer from the result."""

def get_llm():
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=OPENROUTER_API_KEY(),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,  
    )

def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    # Count how many tool results are already in this state
    tool_result_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    if tool_result_count == 0:
        llm = get_llm().bind_tools(TOOLS, tool_choice="required")
    else:
        llm = get_llm().bind_tools(TOOLS)
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "validate"

def validate_node(state: AgentState):
    """Validates the final AI response before sending to user."""
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    issues = []
    # Check for empty or too-short responses
    if not content or len(content.strip()) < 5:
        issues.append("Response is empty or too short.")
    # Check for unresolved tool errors leaked into final answer
    if "error" in content.lower() and len(content) < 80:
        issues.append("Response may contain only an error message.")
    # Check response isn't just raw JSON (tool output leaked)
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            issues.append("Response appears to be raw JSON rather than a human-readable answer.")
        except Exception:
            pass
    if issues:
        return {
            "validation_passed": False,
            "validation_note": " | ".join(issues)
        }
    return {"validation_passed": True, "validation_note": "OK"}

def fix_response_node(state: AgentState):
    """If validation failed, ask LLM to rewrite the response."""
    note = state.get("validation_note", "")
    last = state["messages"][-1]
    fix_prompt = (f"Your previous response had an issue: {note}\n"
                    f"Previous response: {last.content}\n"
                    "Please rewrite it as a helpful, complete, human-readable answer.")
    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"]) + \
                [HumanMessage(content=fix_prompt)]
    response = llm.invoke(messages)
    return {"messages": [response], "validation_passed": True, "validation_note": "Fixed"}

def after_validate(state: AgentState):
    if state.get("validation_passed", True):
        return "end"
    return "fix"

def build_graph():
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent",    agent_node)
    graph.add_node("tools",    tool_node)
    graph.add_node("validate", validate_node)
    graph.add_node("fix",      fix_response_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent",    should_continue, {"tools": "tools", "validate": "validate"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("validate", after_validate,  {"end": END, "fix": "fix"})
    graph.add_edge("fix", END)
    return graph.compile()

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def render_auth_page():
    st.markdown("""
    <style>
    .auth-container { max-width: 420px; margin: 80px auto 0; }
    .auth-title { font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0.3rem; }
    .auth-sub   { text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">🤖 ReAct Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Your AI assistant with live tools</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            with st.form("login_form"):
                uname = st.text_input("Username", placeholder="Enter your username")
                pwd   = st.text_input("Password", type="password", placeholder="Enter your password")
                if st.form_submit_button("Login", use_container_width=True, type="primary"):
                    if uname and pwd:
                        uid = authenticate_user(uname, pwd)
                        if uid:
                            st.session_state.user_id   = uid
                            st.session_state.username  = uname
                            st.session_state.logged_in = True
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password.")
                    else:
                        st.warning("Please fill in both fields.")

        with tab2:
            with st.form("signup_form"):
                new_uname = st.text_input("Choose a username", placeholder="e.g. john_doe")
                new_pwd   = st.text_input("Choose a password", type="password", placeholder="Min 6 characters")
                new_pwd2  = st.text_input("Confirm password",  type="password", placeholder="Repeat password")
                if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                    if not new_uname or not new_pwd:
                        st.warning("Please fill in all fields.")
                    elif len(new_pwd) < 6:
                        st.warning("Password must be at least 6 characters.")
                    elif new_pwd != new_pwd2:
                        st.error("❌ Passwords do not match.")
                    else:
                        uid = create_user(new_uname, new_pwd)
                        if uid:
                            st.success("✅ Account created! Please login.")
                        else:
                            st.error("❌ Username already exists.")
        st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                cid = create_conversation(st.session_state.user_id, "New Chat")
                st.session_state.current_conv_id = cid
                st.session_state.chat_history     = []
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                for k in ["logged_in","user_id","username","current_conv_id","chat_history"]:
                    st.session_state.pop(k, None)
                st.rerun()

        st.divider()
        st.markdown("**💬 Previous Chats**")

        convs = get_conversations(st.session_state.user_id)
        if not convs:
            st.caption("No conversations yet.")
        for cid, title, updated in convs:
            is_active = (cid == st.session_state.get("current_conv_id"))
            col_a, col_b = st.columns([4, 1])
            with col_a:
                label = f"{'▶ ' if is_active else ''}{title}"
                if st.button(label, key=f"conv_{cid}", use_container_width=True,
                            type="primary" if is_active else "secondary"):
                    st.session_state.current_conv_id = cid
                    rows = load_messages(cid)
                    st.session_state.chat_history = [
                        {"role": r, "content": c} for r, c in rows
                    ]
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{cid}", help="Delete this chat"):
                    delete_conversation(cid)
                    if st.session_state.get("current_conv_id") == cid:
                        st.session_state.current_conv_id = None
                        st.session_state.chat_history     = []
                    st.rerun()

def render_chat_page():
    render_sidebar()

    st.title("🤖 ReAct Agent")
    st.caption("Powered by LangGraph · Tools: 🔍 Web · 📰 News · 📈 Stocks · 💱 Currency · 🌤 Weather")

    if not OPENROUTER_API_KEY():
        st.error(
            "⚠️ **OPENROUTER_API_KEY is not set.**\n\n"
            "Add it to your environment or `.streamlit/secrets.toml`:\n"
            "```\nOPENROUTER_API_KEY = \"sk-or-...\"\n```\n"
            "Get a free key at [openrouter.ai](https://openrouter.ai/keys).",
            icon="🔑",
        )
        return

    if not st.session_state.get("current_conv_id"):
        st.info("👈 Click **➕ New Chat** to start a conversation.")
        return

    # Display chat history
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything — stocks, weather, news, currency...")
    if not user_input:
        return

    conv_id = st.session_state.current_conv_id

    # Update title from first message
    rows = load_messages(conv_id)
    if not rows:
        update_conversation_title(conv_id, user_input[:55])

    # Show & save user message
    with st.chat_message("user"):
        st.markdown(user_input)
    save_message(conv_id, "user", user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Build message history for graph
    history_msgs = []
    for m in st.session_state.chat_history[:-1]:  # exclude latest user msg (added below)
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))
    history_msgs.append(HumanMessage(content=user_input))

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and using tools..."):
            try:
                if not OPENROUTER_API_KEY():
                    response_text = ("⚠️ **OpenRouter API key not set.**\n\n"
                                    "Please set `OPENROUTER_API_KEY` in your environment or `st.secrets`.")
                else:
                    graph  = build_graph()
                    result = graph.invoke({
                        "messages": history_msgs,
                        "validation_passed": True,
                        "validation_note": ""
                    })
                    final_msg  = result["messages"][-1]
                    response_text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
                    # Append validation note as footer if something was fixed
                    note = result.get("validation_note", "")
                    if note and note not in ("OK", ""):
                        response_text += f"\n\n---\n*🔍 Validator note: {note}*"
            except Exception as e:
                response_text = f"❌ Agent error: {str(e)}\n\nPlease check your API key and try again."

        st.markdown(response_text)

    save_message(conv_id, "assistant", response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ReAct Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global CSS
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    .stChatMessage { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

    init_db()

    # Session defaults
    for k, v in [("logged_in", False), ("user_id", None), ("username", None),
                ("current_conv_id", None), ("chat_history", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.logged_in:
        render_auth_page()
    else:
        render_chat_page()

if __name__ == "__main__":
    main()