from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from dotenv import load_dotenv
import uuid
import operator

load_dotenv()


model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


checkpointer = MemorySaver()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    name: str
    origin: str
    destination: str
    date: str
    time: str
    price: str

graph=StateGraph(AgentState)



@tool
def book_ticket(name: str, origin: str, destination: str, date: str, time: str, price: str) -> str:    
    """Book a ticket with the provided details."""
    decision=interrupt({"action": "book_ticket", "details": {"name":name,"origin":origin,"destination":destination,"date":date,"time":time,"price":price}})
    
    if isinstance(decision, str) and decision.lower() == "yes":
        return "Ticket Booked Successfully"
    else:
        return "Ticket Booking Failed"

llm_with_tool = model.bind_tools([book_ticket])

def agent(state: AgentState) -> AgentState:
    response = llm_with_tool.invoke(state['messages'])
    return {'messages': [response]}


tool_node=ToolNode([book_ticket])


graph.add_node('agent', agent)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'agent')
graph.add_conditional_edges('agent', tools_condition)
graph.add_edge('tools', 'agent')

agent = graph.compile(checkpointer=checkpointer)