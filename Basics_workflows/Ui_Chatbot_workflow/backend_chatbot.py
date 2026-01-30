from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


checkpointer = MemorySaver()
def chat_node(state: ChatState):

    
    messages = state['messages']

   
    response = llm.invoke(messages)

    
    return {'messages': [response]}

graph = StateGraph(ChatState)


graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)
