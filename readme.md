# 🤖 ReAct Agent

A conversational AI assistant built with **LangGraph**, **LangChain**, and **Streamlit** that uses a ReAct (Reasoning + Acting) loop to answer questions with real-time tool access — no stale training data.

---

## ✨ Features

- 🔍 **Web Search** — Live internet lookups via DuckDuckGo
- 📰 **News** — Latest headlines from Google News RSS (last 24 hours)
- 📈 **Stock Prices** — Real-time stock data via Yahoo Finance
- 💱 **Currency Exchange** — Live FX rates via Frankfurter API 
- 🌤 **Weather** — Current conditions via Open-Meteo 
- 🔐 **User Auth** — Secure login/signup with hashed passwords (SQLite)
- 💬 **Chat History** — Persistent conversations stored per user
- ✅ **Response Validation** — Auto-detects and fixes empty/malformed AI responses

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://python.langchain.com) |
| LLM | OpenAI GPT-4o-mini via [OpenRouter](https://openrouter.ai) |
| Database | SQLite (via `sqlite3`) |
| Web Search | [DuckDuckGo Search (ddgs)](https://pypi.org/project/duckduckgo-search/) |
| News | Google News RSS via `feedparser` |
| Stock Data | Yahoo Finance (free) |
| FX Rates | [Frankfurter API](https://www.frankfurter.app/) (free) |
| Weather | [Open-Meteo](https://open-meteo.com/) (free) |

<img width="233" height="432" alt="graph" src="https://github.com/user-attachments/assets/0d4a1f29-844d-4963-8fd0-9471a5813e30" />


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/react-agent.git
cd react-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

You need an [OpenRouter API key](https://openrouter.ai/keys) (free tier available).

**Option A — `.env` file:**
```env
OPENROUTER_API_KEY=sk-or-...
```

**Option B — Streamlit secrets (`.streamlit/secrets.toml`):**
```toml
OPENROUTER_API_KEY = "sk-or-..."
```

### 4. Run the app

```bash
streamlit run main.py
```

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit
langgraph
langchain
langchain-core
langchain-openai
duckduckgo-search
feedparser
requests
python-dotenv
```

---

## 📁 Project Structure

```
react-agent/
├── main.py           # Main application (agent, tools, UI, database)
├── .env              # API keys (not committed)
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

The agent follows a **ReAct loop** built with LangGraph:

```
User Input → Agent Node → Tool Call → Tool Node → Agent Node → Validate → Response
                                                                    ↓ (if failed)
                                                               Fix Node → Response
```

1. **Agent Node** — On the first turn, the LLM is *forced* to call a tool (no answering from memory). On subsequent turns it decides freely.
2. **Tool Node** — Executes the requested tool and returns results.
3. **Validate Node** — Checks the final response isn't empty, an error, or raw JSON.
4. **Fix Node** — If validation fails, the LLM rewrites the response.

---

## 🔒 Security Notes

- Passwords are hashed with SHA-256 before storage.
- API keys are read from environment variables or Streamlit secrets — never hardcoded.
- The `chat_agent.db` SQLite file is created locally and should be added to `.gitignore`.

### Recommended `.gitignore`

```
.env
*.db
__pycache__/
.streamlit/secrets.toml
```

---

## 🌐 Deployment

This app can be deployed to **[Streamlit Community Cloud](https://streamlit.io/cloud)** for free:

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Add `OPENROUTER_API_KEY` under **App settings → Secrets**.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.


