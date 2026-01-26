from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import  TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

class StoryState(TypedDict):
    topic: str
    title: str
    description: str
    story: str
    summary: str
    

def title_write(state: StoryState) -> StoryState:
    prompt=f"Write a title for the story: {state['topic']}"
    title = model.invoke(prompt).content
    state['title'] = title
    return state

def description_write(state: StoryState) -> StoryState:
    prompt=f"Write a description for the story: {state['topic']}"
    description = model.invoke(prompt).content
    state['description'] = description
    return state

def story_write(state: StoryState) -> StoryState:
    prompt=f"Write the story: {state['title']} {state['description']}"
    story = model.invoke(prompt).content
    state['story'] = story
    return state

def summary_write(state: StoryState) -> StoryState:
    prompt=f"Write a summary for the story: {state['story']}"
    summary = model.invoke(prompt).content
    state['summary'] = summary
    return state


graph=StateGraph(StoryState)    


graph.add_node('title', title_write)
graph.add_node('description', description_write)
graph.add_node('story', story_write)
graph.add_node('summary', summary_write)

graph.add_edge(START, 'title')
graph.add_edge('title', 'description')
graph.add_edge('description', 'story')
graph.add_edge('story', 'summary')
graph.add_edge('summary', END)

workflow = graph.compile()

state = {'topic': 'university ki larki pasandh ha uska name gull meena ha or larka ka name dameer laken larki ko nahi pata larka usa pasandh karta ha romon urdu ma title'} 

output = workflow.invoke(state)

print(f"Topic: {output['topic']} \nTitle: {output['title']} \nDescription: {output['description']} \nStory: {output['story']} \nSummary: {output['summary']}")