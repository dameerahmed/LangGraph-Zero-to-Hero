from langgraph.graph import StateGraph, START, END
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
def chat_node(state: ChatState):

    
    messages = state['messages']

   
    response = llm.invoke(messages)

    
    return {'messages': [response]}

graph = StateGraph(ChatState)


graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

