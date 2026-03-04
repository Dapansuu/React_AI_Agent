import sqlite3
import os
import requests
import feedparser
import yfinance as yf

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    AIMessage
)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from ddgs import DDGS
from sympy import sympify, pi, E
from sympy.core.sympify import SympifyError

#* ENV 
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# *LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


#* TOOLS - Search, Weather, Calculator, Stock Price, News, Currency Converter

def search_tool(query: str):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return list(results)


@tool
def search(query: str) -> str:
    """Search the web for current information."""
    results = search_tool(query)

    formatted = []
    for r in results:
        formatted.append(
            f"Title: {r.get('title')}\n"
            f"Snippet: {r.get('body')}\n"
            f"Link: {r.get('href')}\n"
        )

    return "\n\n".join(formatted)


@tool
def weather(city: str) -> str:
    """Get current weather for a city."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1}
    geo_response = requests.get(geo_url, params=geo_params).json()

    if "results" not in geo_response:
        return "City not found."

    lat = geo_response["results"][0]["latitude"]
    lon = geo_response["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    weather_response = requests.get(weather_url, params=weather_params).json()

    temp = weather_response["current_weather"]["temperature"]
    wind = weather_response["current_weather"]["windspeed"]

    return f"Temperature: {temp}°C\nWind Speed: {wind} km/h"


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = sympify(expression, locals={"pi": pi, "E": E})
        return str(result.evalf())
    except SympifyError:
        return "Invalid mathematical expression."


@tool
def stock_price(ticker: str) -> str:
    """Get latest stock price."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            return "Stock not found."

        price = data["Close"].iloc[-1]
        return f"{ticker} current price: ${price:.2f}"
    except Exception:
        return "Error retrieving stock data."


@tool
def news(topic: str) -> str:
    """Get latest news headlines."""
    try:
        url = f"https://news.google.com/rss/search?q={topic}"
        feed = feedparser.parse(url)

        headlines = []
        for entry in feed.entries[:5]:
            headlines.append(
                f"Title: {entry.title}\nLink: {entry.link}"
            )

        return "\n\n".join(headlines)
    except Exception:
        return "Error fetching news."


@tool
def currency_converter(query: str) -> str:
    """Convert currency. Format: 100 USD to INR"""
    try:
        parts = query.split()
        amount = float(parts[0])
        from_currency = parts[1]
        to_currency = parts[3]

        url = "https://api.exchangerate.host/convert"
        params = {
            "from": from_currency,
            "to": to_currency,
            "amount": amount
        }

        response = requests.get(url, params=params).json()

        if "result" not in response:
            return "Conversion failed."

        return f"{amount} {from_currency} = {response['result']:.2f} {to_currency}"

    except Exception:
        return "Invalid format. Use: 100 USD to INR"


tools = [search, weather, calculator, stock_price, news, currency_converter]

tool_node = ToolNode(tools)


llm_with_tools = llm.bind_tools(tools)

from langchain_core.runnables import RunnableLambda

system_prompt = SystemMessage(
    content=(
        "You are a strict fact-checking assistant.\n"
        "For politics, current events, office holders, "
        "ALWAYS use the search tool first.\n"
        "Never answer from memory.\n"
    )
)

def add_system(messages):
    # Only add system if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        return [system_prompt] + messages
    return messages

llm_with_tools = RunnableLambda(add_system) | llm_with_tools

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int

def reasoning_node(state: ChatState):
    if state["steps"] > 8:
        return {"messages": [AIMessage(content="Max steps reached.")]}
    response = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [response]}

def router(state: ChatState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "final"


conn = sqlite3.connect("react.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node("reasoning", reasoning_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "reasoning")
graph.add_conditional_edges(
    "reasoning",
    router,
    {
        "tool_node": "tool_node",
        "final": END
    }
)
graph.add_edge("tool_node", "reasoning")

chatbot = graph.compile(checkpointer=checkpointer)


# CLI LOOP


# if __name__ == "__main__":

#     config = {"configurable": {"thread_id": "cli-session-1"}}

#     print("🤖 Agent started. Type 'exit' to quit.\n")

#     while True:
#         user_input = input("You: ")

#         if user_input.lower() == "exit":
#             break

#         result = chatbot.invoke(
#             {"messages": [HumanMessage(content=user_input)]},
#             config=config
#         )

#         final_message = result["messages"][-1]
#         print("\nAgent:", final_message.content)