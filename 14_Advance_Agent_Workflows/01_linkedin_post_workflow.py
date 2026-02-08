from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from dotenv import load_dotenv
from pathlib import Path
import operator

load_dotenv()

# --- Data Models ---
class Task(BaseModel):
    id: int
    title: str = Field(..., description="The title of the task.")
    description: str = Field(..., description="The description of the task.")

class Plan(BaseModel):
    post_title: str 
    tasks: List[Task]

# --- State ---
class State(TypedDict):
    topic: str
    plan: Plan
    # Sorting ke liye list of dicts
    sections: Annotated[List[dict], operator.add]
    final: str

# --- LLM Setup ---
gemini_model = ChatGroq(
    model="qwen/qwen3-32b", 
    temperature=0.7
)

# --- Nodes ---

def orchestrator(state: State) -> dict:
    plan = gemini_model.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content="Generate a plan with 5-7 sections on the following topic."
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {"plan": plan}

def fanout(state: State):
    return [
        Send(
            "worker",
            {"task": task, "plan": state["plan"], "topic": state["topic"]}
        ) for task in state["plan"].tasks
    ]

def worker(payload: dict) -> dict:
    task = payload["task"]
    plan = payload["plan"]
    topic = payload["topic"]
    
    post_title = plan.post_title
    
    section_md = gemini_model.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(content=(
                f"Post: {post_title}\n\n"
                f"Topic: {topic}\n\n"
                f"Section: {task.title}\n\n"
                f"Description: {task.description}\n\n"
                "Return only the section content in Markdown."
            )),
        ]
    ).content.strip()
    
    # Returning ID for sorting
    return {"sections": [{"id": task.id, "content": section_md}]}

def reducer(state: State) -> State:
    title = state["plan"].post_title
    
    # 1. Sorting Logic
    sorted_sections = sorted(state["sections"], key=lambda x: x['id'])
    body = "\n\n".join([s["content"] for s in sorted_sections]).strip()
    
    final_md = f"# {title}\n\n{body}\n"
    
    # --- FIX START: Filename Cleaning ---
    # Windows mein invalid characters (: ? * " < > |) remove kar rahe hain
    clean_filename = "".join(c for c in title if c.isalnum() or c in " _-")
    filename = clean_filename.strip().lower().replace(" ", "_") + ".md"
    # --- FIX END ---

    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")
    
    # Terminal mein batayega ke file kahan save hui
    print(f"✅ File Saved Successfully: {output_path.absolute()}")
    
    return {"final": final_md}

# --- Graph ---
graph = StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

workflow = graph.compile()

# --- Execution ---
print("Running Workflow...")
response = workflow.invoke({"topic": "Future of Agentic AI"})