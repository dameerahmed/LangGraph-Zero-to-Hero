from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)
class state(TypedDict):
    topic:str
    joke:str
    explanation:str
    
    
graph = StateGraph(state)


def joke_write(state: state) -> state:
    prompt=f"Write a joke about {state['topic']}"
    joke = model.invoke(prompt).content
    return {"joke": joke}

def explanation_write(state: state) -> state:
    prompt=f"Write an explanation for the joke: {state['joke']}"
    explanation = model.invoke(prompt).content
    return {"explanation": explanation}
    



graph.add_node('joke', joke_write)
graph.add_node('explanation', explanation_write)



graph.add_edge(START, 'joke')
graph.add_edge('joke', 'explanation')
graph.add_edge('explanation', END)

checkpointer=InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

state = {'topic': 'pakistan'} 


config1={
    'configurable':{'thread_id':'1'},
}
output = workflow.invoke(state,config=config1)
st=workflow.get_state_history(config1)
print(f"Topic: {output['topic']} \nJoke: {output['joke']} \nExplanation: {output['explanation']}")

for s in st:
    print(s)