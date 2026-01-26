from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import  TypedDict,Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


class review(BaseModel):
    review: Literal["positive", "negative"] = Field(
        description="Review of the user input",
    )

structure_model=model.with_structured_output(review)
class Chatbot(TypedDict):
    feedback: str
    review: review
    answer: str
    

graph = StateGraph(Chatbot)


def llmq(state: Chatbot) -> Chatbot:
    question = state['feedback']
    prompt=f"Answer the question: {question} in positive or negative"
    answer = structure_model.invoke(prompt).review
    return {"review": answer}

def positive(state: Chatbot) -> Chatbot:
    review = state['review']
    print("calling positive node")
    prompt=f"write a positive feedback about the review: {state['feedback']} in 50 words"
    answer = model.invoke(prompt).content
    return {"answer": answer}

def negative(state: Chatbot) -> Chatbot:
    review = state['review']
    print("calling negative node")
    prompt=f"write a negative feedback about the review: {state['feedback']} in 50 words"
    answer = model.invoke(prompt).content
    return {"answer": answer}

def condition(state: Chatbot) -> Chatbot:
    if state['review'] == 'positive':
        return 'positive'
    elif state['review'] == 'negative':
        return 'negative'



graph.add_node('llm', llmq)
graph.add_node('positive', positive)
graph.add_node('negative', negative)

graph.add_edge(START, 'llm')
graph.add_conditional_edges(
    'llm', 
    condition,
)

graph.add_edge('positive', END)
graph.add_edge('negative', END)

workflow = graph.compile()

output = workflow.invoke({
    'feedback': "I dont like this product."})

print(f"Feedback: {output['feedback']} \nReview: {output['review']} \nAnswer: {output['answer']}")
    
        
    
    
   

