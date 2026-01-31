import streamlit as st
from backend import chatbot, generate_thread_id, generate_chat_title
from langchain_core.messages import HumanMessage, BaseMessage
import json
import os

THREADS_FILE = "threads.json"
# utils funtions

def load_threads():
    if os.path.exists(THREADS_FILE):
        with open(THREADS_FILE, "r") as f:
            return json.load(f)
    return []

def save_threads(threads):
    with open(THREADS_FILE, "w") as f:
        json.dump(threads, f)
def add_thread_id(thread_id, title="New Chat"):    
    if not any(t['id'] == thread_id for t in st.session_state.chat_thread):
        st.session_state.chat_thread.append({'id': thread_id, 'title': title})
        save_threads(st.session_state.chat_thread)
def reset_chat():
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history.clear() 
    add_thread_id(st.session_state.thread_id)
    st.rerun()

def load_conversation(thread_id):
    state= chatbot.get_state( config={"configurable": {"thread_id":thread_id}})
    if state and "messages" in state.values:
        return state.values["messages"]
    return []

# 0. Page Config
st.set_page_config(page_title="Gemini", layout="centered")
st.title("🤖 Gemini ")


    

# 1. Session State Initialize
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = load_threads()

add_thread_id(st.session_state.thread_id)    
    
    
# 2 sidebar

st.sidebar.title("🤖 Gemini ")
st.sidebar.write("Powered by LangGraph")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("conversation History")
for thread in st.session_state.chat_thread[::-1]:
    if thread['title'] != None and thread['title'] != "New Chat":
        if st.sidebar.button(thread['title'], key=thread['id']):
            st.session_state.thread_id = thread['id']
            history = load_conversation(thread['id'])
        
            tem=[]
            for message in history:
                if isinstance(message, HumanMessage):
                    tem.append({"role": "user", "content": message.content})
                else:
                    tem.append({"role": "assistant", "content": message.content})
            st.session_state.message_history = tem
# 3. history
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input Handle karna
if user_input := st.chat_input("Sawal pucho..."):
    config={"configurable": {"thread_id": st.session_state.thread_id}}
    # User ka message screen par dikhana aur history mein save karna
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 4. Assistant ka Response aur Streaming
    with st.chat_message("assistant"):
        
        def response_generator():
            full_content = ""
            
            # LangGraph Stream call
            # Hum 'messages' mode use kar rahe hain for token-by-token streaming
            events = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )

            for chunk, metadata in events:
                # Sirf tab yield karo jab chunk mein text content ho
                if isinstance(chunk, BaseMessage) and chunk.content:
                    content = chunk.content
                    full_content += content
                    yield content
                elif isinstance(chunk, str): # Kabhi kabhi chunks direct string hote hain
                    full_content += chunk
                    yield chunk
            
            return full_content

        
        try:
            ai_message = st.write_stream(response_generator())
            
            st.session_state.message_history.append({"role": "assistant", "content": ai_message})
            
            for thread in st.session_state.chat_thread:
                if thread['id'] == st.session_state.thread_id:
                    if thread['title'] == None or thread['title'] == "New Chat":
                        thread['title'] = generate_chat_title(user_input, ai_message)
                        save_threads(st.session_state.chat_thread)
                        st.rerun()
        except Exception as e:
            st.error(f"Streaming mein masla aaya: {e}")