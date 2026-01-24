from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing import  TypedDict
from dotenv import load_dotenv

load_dotenv()

Gimi_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

Groq_model = ChatGroq(
    model="qwen/qwen3-32b",
)

class LegalState(TypedDict):
    case_details: str
    
    # Names 
    prosecution_name: str  
    defense_name: str       

    # Statements (Bayan)
    A_witness_statement: str  
    B_witness_statement: str  

    # Arguments (Dono pehlu: Support + Attack)
    prosecutor_argument: str 
    defense_argument: str    

    # Faisla
    final_verdict: str

def prosecutor_node(state: LegalState):
    prompt = f"""
    ROLE: You are the Prosecutor representing {state['prosecution_name']}.
    
    CASE: {state['case_details']}
    
    YOUR CLIENT ({state['prosecution_name']}'s Statement): 
    "{state['A_witness_statement']}"
    
    THE ACCUSED ({state['defense_name']}'s Statement): 
    "{state['B_witness_statement']}"
    
    TASK: Write a legal argument in two strict sections:
    
    SECTION 1: VALIDATION (Why {state['prosecution_name']} is right)
    - Explain why your client's statement matches the facts.
    - Highlight their honesty.
    
    SECTION 2: ATTACK (Why {state['defense_name']} is lying)
    - Find loopholes in the accused's statement.
    - Prove they are guilty and trying to hide the truth.
    
    Tone: Aggressive, logical, and authoritative language romon urdu maximum 150 words.
    """
    
    
    response = Groq_model.invoke(prompt).content
    return {"prosecutor_argument": response}

def defense_node(state: LegalState):
    prompt = f"""
    ROLE: You are the Defense Attorney representing {state['defense_name']}.
    
    CASE: {state['case_details']}
    
    THE ACCUSER ({state['prosecution_name']}'s Statement): 
    "{state['A_witness_statement']}"
    
    YOUR CLIENT ({state['defense_name']}'s Statement): 
    "{state['B_witness_statement']}"
    
    TASK: Write a legal argument in two strict sections:
    
    SECTION 1: DEFENSE (Why {state['defense_name']} is innocent)
    - Validate your client's alibi or reasoning.
    - Explain why they had no motive.
    
    SECTION 2: DISCREDIT (Why {state['prosecution_name']} is wrong)
    - Attack the accuser's credibility.
    - Show how they are confused or lying to frame your client.
    
    Tone: Protective, sharp, and persuasive  language romon urdu  maximum 150 words.
    """
    
    
    response = Groq_model.invoke(prompt).content
    return {"defense_argument": response}

def judge_node(state: LegalState):
    prompt = f"""
    ROLE: You are the Supreme Judge.
    
    CASE: {state['case_details']}
    
    PROSECUTOR'S ARGUMENT (Representing {state['prosecution_name']}):
    {state['prosecutor_argument']}
    
    DEFENSE'S ARGUMENT (Representing {state['defense_name']}):
    {state['defense_argument']}
    
    TASK:
    1. Analyze the 'Attack' points from both sides.
    2. Analyze the 'Validation' points from both sides.
    3. Decide who is telling the truth: {state['prosecution_name']} or {state['defense_name']}?
    4.  language romon urdu
    5. maximum 150 words
    FINAL VERDICT: Give a clear ruling with a reason.
    """
    
    verdict = Gimi_model.invoke(prompt).content
    return {"final_verdict": verdict}



graph = StateGraph(LegalState)

graph.add_node('prosecutor', prosecutor_node)
graph.add_node('defense', defense_node)
graph.add_node('judge', judge_node)

graph.add_edge(START, 'prosecutor')
graph.add_edge(START, 'defense')
graph.add_edge('prosecutor', 'judge')
graph.add_edge('defense', 'judge')
graph.add_edge('judge', END)


workflow = graph.compile()


inputs = {
    "case_details": "Office mein laptop chori hua hai dopehar 2 bajey.",
    
    "prosecution_name": "Manager Kashif",
    "A_witness_statement": "Maine Sameer ko table ke paas ghabraye hue dekha tha jab main wahan se guzra.",
    
    "defense_name": "Intern Sameer",
    "B_witness_statement": "Main wahan sirf pani peenay gaya tha, laptop pehle hi gayab tha."
}

result = workflow.invoke(inputs)

print(f"--- Prosecutor ({inputs['prosecution_name']}) ---")
print(result['prosecutor_argument'])
print(f"\n--- Defense ({inputs['defense_name']}) ---")
print(result['defense_argument'])
print("\n--- JUDGE'S DECISION ---")
print(result['final_verdict'])

def save_graph_image():
    print("Graph image generate ho rahi hai...")
    try:
        # Graph ka data bytes mein lein
        graph_image = workflow.get_graph().draw_mermaid_png()
        
        # File save karein
        with open("legal_case_workflow.png", "wb") as f:
            f.write(graph_image)
            
        print("✅ Graph save ho gaya! Check karein: 'legal_case_workflow.png'")
        
    except Exception as e:
        print(f"❌ Graph nahi ban saka. Error: {e}")

# Function call karein
save_graph_image()