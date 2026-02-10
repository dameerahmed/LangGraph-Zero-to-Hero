import os
import re
import time
from typing import List, TypedDict
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# 1. Setup Retrieval System (Fixed for Rate Limits)
# -----------------------------

# Try loading the PDF
try:
    print("Loading PDF...")
    loader = PyPDFLoader("PYTHON PROGRAMMING NOTES.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")
except Exception as e:
    print(f"Could not load PDF: {e}")
    docs = []

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs[::-20])

# Clean encoding
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

# --- MODEL SELECTION ---
# Wapis embedding-001 use kar rahe hain kyunki ye available hai
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- BATCHING LOGIC (CRITICAL FIX) ---
def create_vector_store_slowly(chunks, embeddings, batch_size=10):
    vector_store = None
    total = len(chunks)
    print(f"⚠️ Total chunks to process: {total}")
    print("⏳ Starting slow embedding to avoid Rate Limit (this will take time)...")
    
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"   - Embedding batch {i} to {min(i + batch_size, total)}...")
        
        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            
            # Har batch ke baad 3 second ka sukoon (Wait)
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
            print("   Waiting 20 seconds before retrying...")
            time.sleep(20)
            # Retry once
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

    print("✅ Vector Store Created Successfully!")
    return vector_store

# Create Vector Store safely
if chunks:
    vector_store = create_vector_store_slowly(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
else:
    print("No chunks to embed.")
    retriever = None

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Thresholds
UPPER_TH = 0.7
LOWER_TH = 0.3

# -----------------------------
# 2. State Definition
# -----------------------------
class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_docs: List[Document]
    answer: str

# -----------------------------
# 3. Nodes
# -----------------------------

def retrieve_node(state: State) -> State:
    print("\n---RETRIEVING---")
    if retriever:
        return {"docs": retriever.invoke(state["question"])}
    return {"docs": []}

class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a retrieval evaluator. Score relevance [0.0, 1.0]. Output JSON."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}")
    ]) 
    | llm.with_structured_output(DocEvalScore)
)

def eval_each_doc_node(state: State) -> State:
    print("---EVALUATING DOCS---")
    scores = []
    good = []
    
    for d in state.get("docs", []):
        try:
            out = doc_eval_chain.invoke({"question": state["question"], "chunk": d.page_content})
            if out.score > LOWER_TH:
                good.append(d)
            scores.append(out.score)
        except:
            pass

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": "Good chunks found."}
    
    if not scores or all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "No good chunks."}
        
    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Mixed signals."}

tavily = TavilySearchResults(max_results=5)

def web_search_node(state: State) -> State:
    print("---WEB SEARCH---")
    try:
        results = tavily.invoke({"query": state["question"]})
        web_docs = [
            Document(page_content=f"TITLE: {r['title']}\nCONTENT: {r['content']}", metadata={"url": r['url']})
            for r in results
        ]
        return {"web_docs": web_docs}
    except:
        return {"web_docs": []}

class KeepOrDrop(BaseModel):
    keep: bool

filter_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Filter irrelevant sentences. Return keep=true if helpful."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}")
    ])
    | llm.with_structured_output(KeepOrDrop)
)

def refine(state: State) -> State:
    print("---REFINING---")
    # Select docs based on verdict
    docs = state["good_docs"] if state["verdict"] == "CORRECT" else state.get("web_docs", [])
    
    full_text = " ".join([d.page_content for d in docs]).replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    
    kept = []
    # Limit to first 15 sentences to save time/tokens
    for s in sentences[:15]: 
        if len(s) > 20:
            try:
                if filter_chain.invoke({"question": state["question"], "sentence": s}).keep:
                    kept.append(s)
            except: pass
            
    return {"refined_context": " ".join(kept)}

def generate(state: State) -> State:
    print("---GENERATING---")
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer using context only."),
            ("human", "Question: {question}\n\nContext:\n{refined_context}")
        ]) 
        | llm
    )
    return {"answer": chain.invoke({"question": state["question"], "refined_context": state["refined_context"]}).content}

def ambiguous_node(state: State) -> State:
    return {"answer": "Ambiguous query, could not decide."}

def router(state: State):
    v = state["verdict"]
    if v == "CORRECT": return "refine"
    if v == "INCORRECT": return "web_search"
    return "ambiguous"

# -----------------------------
# 4. Graph Build
# -----------------------------
g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)
g.add_node("ambiguous", ambiguous_node)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges("eval_each_doc", router, {"refine": "refine", "web_search": "web_search", "ambiguous": "ambiguous"})
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)
g.add_edge("ambiguous", END)

app = g.compile()

# -----------------------------
# 5. Run
# -----------------------------
print("🚀 Starting Workflow...")
res = app.invoke({"question": "What is Python?", "docs": []})
print("\nFINAL ANSWER:\n", res["answer"])