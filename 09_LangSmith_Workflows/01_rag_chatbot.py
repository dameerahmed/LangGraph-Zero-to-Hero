import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

# Configuration
PDF_PATH = "C:/Users/Dameer Ahmed/Documents/Development/langChain_practice/Introduction_to_Python_Programming_-_WEB.pdf"

def main():
    # 1. Load PDF (Limited to 5 pages for speed and to avoid rate limits)
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()[:5] 
    
    # 2. Split Text
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    # 3. Embed and Store (Simple FAISS)
    print(f"Embedding {len(splits)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 4. Define Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Chat Loop
    print("\n✅ RAG Chatbot Ready! (Type 'exit' to quit)")
    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        try:
            response = rag_chain.invoke(question)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()