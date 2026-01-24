from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import  TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

class Chatbot(TypedDict):
    question: str
    answer: str


def llmq(state: Chatbot) -> Chatbot:
    question = state['question']
    prompt=f"Answer the question: {question}"
    answer = model.invoke(prompt).content
    state['answer'] = answer
    return state



graph = StateGraph(Chatbot)

graph.add_node('llm', llmq)

graph.add_edge(START, 'llm')
graph.add_edge('llm', END)

workflow = graph.compile()

output = workflow.invoke({
    'question': "What is the capital of France?"})

print(f"Question: {output['question']} \nAnswer: {output['answer']}")