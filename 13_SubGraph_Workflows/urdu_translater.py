from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing import  TypedDict
from dotenv import load_dotenv

load_dotenv()

subgraph_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

parent_model = ChatGroq(
    model="qwen/qwen3-32b",
)


class substate(TypedDict):
    text: str
    urdu_text: str


class parentstate(TypedDict):
    question: str
    answer: str
    urdu_answer: str
    

subgraph=StateGraph(substate)


def urdu_translater(state: substate)->substate:
    prompt = f"Translate the following text into Roman Urdu: {state['text']}"
    urdu_text = subgraph_model.invoke(prompt).content
    return {"urdu_text": urdu_text}

subgraph.add_node('llm', urdu_translater)

subgraph.add_edge(START, 'llm')
subgraph.add_edge('llm', END) 

subgraph = subgraph.compile()           



parentgraph=StateGraph(parentstate)


def answer_generator(state: parentstate)->parentstate:
    prompt=f"Answer the question in 2 lines: {state['question']}"
    answer = parent_model.invoke(prompt).content
    return {"answer": answer}

def translate_answer(state: parentstate)->parentstate:
    response = subgraph.invoke({"text": state['answer']})
    return {"urdu_answer": response["urdu_text"]}

 
parentgraph.add_node('answer_generator', answer_generator)
parentgraph.add_node('translate_answer', translate_answer)

parentgraph.add_edge(START, 'answer_generator')
parentgraph.add_edge('answer_generator', 'translate_answer')
parentgraph.add_edge('translate_answer', END)

parentgraph = parentgraph.compile()


output = parentgraph.invoke({
    'question': "What is the capital of Pakistan?"
})


print(f"Question: {output['question']} \nAnswer: {output['answer']} \nUrdu Answer: {output['urdu_answer']}")