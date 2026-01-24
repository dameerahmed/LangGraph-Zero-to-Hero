from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing import  TypedDict,Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator


load_dotenv()

Gimi_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

Groq_model = ChatGroq(
    model="qwen/qwen3-32b",
)


class output(BaseModel):
    feedback: str = Field(
        description="Feedback about the essay ",
    )
    score: int = Field(
        description="Score of the essay",
        ge=0,
        le=10
    )
    
structure_model=Groq_model.with_structured_output(output)    

class essaystate(TypedDict):
    essay: str
    anlysis_feedback: str
    language_feedback: str 
    structure_feedback: str
    summary_feedback: str
    total_score:  Annotated[list[int],operator.add]  
    final_score: int


    
    
    
graph=StateGraph(essaystate)    


def essay_write(state:essaystate) -> essaystate:
    prompt=f"write an essay on pakistan in 200 words"
    essay=Groq_model.invoke(prompt).content
    return {"essay": essay}

def analysis_feedback(state:essaystate) -> essaystate:
    prompt=f"write a feedback about the analysis of the essay: {state['essay']} in 50 words and give a score out of 10"
    output=structure_model.invoke(prompt)
    return {"anlysis_feedback": output.feedback, "total_score": [output.score]}

def language_feedback(state:essaystate) -> essaystate:
    prompt=f"write a feedback about the language of the essay: {state['essay']} in 50 words and give a score out of 10"
    output=structure_model.invoke(prompt)
    return {"language_feedback": output.feedback, "total_score": [output.score]}

def structure_feedback(state:essaystate) -> essaystate:
    prompt=f"write a feedback about the structure of the essay: {state['essay']} in 50 words and give a score out of 10"
    output=structure_model.invoke(prompt)
    return {"structure_feedback": output.feedback, "total_score": [output.score]}

def summary_feedback(state:essaystate) -> essaystate:
    prompt=f"write a summary feedback about the {state['anlysis_feedback']} {state['language_feedback']} {state['structure_feedback']} in 50 words"
    overall_feedback=Gimi_model.invoke(prompt).content
    avr_score=sum(state['total_score'])/len(state['total_score'])
    return {'summary_feedback': overall_feedback, 'final_score': avr_score}


graph.add_node('essay', essay_write)
graph.add_node('analysis', analysis_feedback)
graph.add_node('language', language_feedback)
graph.add_node('structure', structure_feedback)
graph.add_node('summary', summary_feedback)    


graph.add_edge(START, 'essay')
graph.add_edge('essay', 'analysis')
graph.add_edge('essay', 'language')
graph.add_edge('essay', 'structure')
graph.add_edge('analysis', 'summary')
graph.add_edge('language', 'summary')
graph.add_edge('structure', 'summary')
graph.add_edge('summary', END)

workflow = graph.compile()

result=workflow.invoke({})

    
print("Essay : ",result['essay'])    
print("Analysis Feedback : ",result['anlysis_feedback'])
print("Language Feedback : ",result['language_feedback'])
print("Structure Feedback : ",result['structure_feedback'])
print("Summary Feedback : ",result['summary_feedback'])
print("Total Score : ",result['total_score'])
print("Final Score : ",result['final_score'])
