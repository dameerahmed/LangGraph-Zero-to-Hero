from langgraph.graph import StateGraph, START, END
from typing import  TypedDict
from dotenv import load_dotenv
import random

load_dotenv()


class dbconnect(TypedDict):
    status: str
    round: int
    


graph = StateGraph(dbconnect)


def connect(state: dbconnect) -> dbconnect:
    connect=random.choice(['success', 'failure'])
    if connect == 'success':
        return {'status': 'success', 'round': state['round'] + 1}
    else:
        return {'status': 'failure', 'round': state['round'] + 1}


def condition(state: dbconnect) -> dbconnect:
    if state['status'] == 'success':
        return 'success'
    if state['status'] == 'failure':
        return 'connect'  
    if state['round'] == 3:
        return 'failure'  

graph.add_node('connect', connect)


graph.add_edge(START, 'connect')
graph.add_conditional_edges(
    'connect', 
    condition, 
    {
        'success': END,
        'failure': END,
        'connect': 'connect',
    },
)





workflow = graph.compile()

output = workflow.invoke({'round': 0})

print(f"Status: {output['status']} \nRound: {output['round']}")
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