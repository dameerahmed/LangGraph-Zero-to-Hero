from langgraph.graph import StateGraph, START, END

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import uuid

load_dotenv()
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

@tool
def calculator(num1:float,num2:float,op:str)->float:
    """any calculation you want to do"""
    if op == '+':
        return num1+num2
    elif op == '-':
        return num1-num2
    elif op == '*':
        return num1*num2
    elif op == '/':
        return num1/num2
    
@tool
def info()->str:
    """A tool for getting info about dameer ."""
    return "Hello, I'm Dameer, iam curent student data science queida awam university father nam eshahid ali"

search_tool = DuckDuckGoSearchRun(region="in-en")

tools=[calculator,info,search_tool]
llm_With_tools=llm.bind_tools(tools)


conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

def generate_thread_id():
    thread_id=str(uuid.uuid4())
    return thread_id
def generate_chat_title(user_input, assistant_response):
    
    prompt = f"""
    Task: Create a 2-4 word concise title for a chat.
    Rules: 
    - Do not use quotes, periods, or the word 'Title'.
    - Focus on the main topic discussed.
    - Respond ONLY with the title.

    User said: {user_input}
    Assistant replied: {assistant_response}
    """
    
    summary = llm.invoke([HumanMessage(content=prompt)])
    return summary.content.strip()


tool_node=ToolNode(tools)

def chat_node(state: ChatState):

    
    messages = state['messages']

   
    response = llm_With_tools.invoke(messages)

    
    return {'messages': [response]}

graph = StateGraph(ChatState)


graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpointer)

