from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langgraph.graph import StateGraph,START,END
import requests
from typing import TypedDict,Annotated
from langgraph.graph.message import BaseMessage,add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun,tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_ollama import ChatOllama
from pydantic import Field
import os

load_dotenv()
class chatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages] = Field(description='''You are a helpful financial assistant.
When you receive raw data (JSON/Dictionaries), analyze it and generate a human-readable response.
CRITICAL: DO NOT output, print, or repeat the raw JSON data or dictionary in your response.
Start your response directly with the natural language summary.
''')
vantage_api_key = os.getenv("vantage_api_key")
model = ChatOllama(model="qwen2.5:7b",num_ctx=2048)
@tool
def calculator(first_num:float,second_num:float,operation:str)-> str:
    '''Performs basic arithmetic operations. Takes "add","subtract","multilply","divide"
    as operation, first_num and second_num as operands'''
    if operation == "add":
        return str(first_num + second_num)
    elif operation == "subtract":
        return str(first_num - second_num)
    elif operation == "multiply":
        return str(first_num * second_num)
    elif operation == "divide":
        return str(first_num / second_num)
    else:
        return "Invalid operation"
@tool
def getStockPrice(ticker:str):
    """
    Fetches stock data. 
    IMPORTANT: Input must be a TICKER SYMBOL (e.g., 'AAPL', 'TSLA', 'IBM'), NOT a company name.
    """
    # Force uppercase just in case the AI sends 'tsla'
    ticker = ticker.upper()
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={vantage_api_key}"
    return requests.get(url).json()['Time Series (Daily)']['2026-02-03']
search = DuckDuckGoSearchRun(region="US")
tools = [calculator,getStockPrice,search]
model_withtools = model.bind_tools(tools=tools)
tool_node = ToolNode(tools=tools)

def chatNode(state: chatState):
    messages = state['messages']
    response = model_withtools.invoke(messages)
    return{'messages':[response]}

graph = StateGraph(chatState) 
graph.add_node('chatnode',chatNode)
graph.add_node('tools',tool_node)

graph.add_edge(START,"chatnode")
graph.add_conditional_edges("chatnode",tools_condition)
graph.add_edge("tools","chatnode")
checkpoint = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpoint)

