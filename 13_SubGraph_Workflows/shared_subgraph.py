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




class parentstate(TypedDict):
    question: str
    answer: str
    urdu_answer: str
    

subgraph=StateGraph(parentstate)


def urdu_translater(state: parentstate)->parentstate:
    prompt = f"Translate the following text into Roman Urdu: {state['answer']}"
    urdu_text = subgraph_model.invoke(prompt).content
    return {"urdu_answer": urdu_text}

subgraph.add_node('llm', urdu_translater)

subgraph.add_edge(START, 'llm')
subgraph.add_edge('llm', END) 

subgraph = subgraph.compile()           



parentgraph=StateGraph(parentstate)


def answer_generator(state: parentstate)->parentstate:
    prompt=f"Answer the question in 2 lines: {state['question']}"
    answer = parent_model.invoke(prompt).content
    return {"answer": answer}


 
parentgraph.add_node('answer_generator', answer_generator)
parentgraph.add_node('translate_answer', subgraph)

parentgraph.add_edge(START, 'answer_generator')
parentgraph.add_edge('answer_generator', 'translate_answer')
parentgraph.add_edge('translate_answer', END)

parentgraph = parentgraph.compile()


output = parentgraph.invoke({
    'question': "What is the capital of Pakistan?"
})


print(f"Question: {output['question']} \nAnswer: {output['answer']} \nUrdu Answer: {output['urdu_answer']}")