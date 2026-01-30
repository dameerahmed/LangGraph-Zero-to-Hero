import streamlit as st
from backend_chatbot import chatbot
from langchain_core.messages import HumanMessage
import uuid
import time

st.set_page_config(page_title="Gemini", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500&display=swap');
    
    .stApp {
        background-color: #131314 !important;
        color: #e3e3e3 !important;
        font-family: 'Google Sans', sans-serif !important;
    }

    [data-testid="stHeader"], header, footer { visibility: hidden; height: 0px !important; }
    div[data-testid="stToolbar"] { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        width: 280px !important;
        border: none !important;
    }
    
    .stButton > button {
        background-color: #1a1c1e !important;
        color: #e3e3e3 !important;
        border: 1px solid #444746 !important;
        border-radius: 12px !important;
        height: 48px !important;
        width: 100% !important;
        text-align: left !important;
        padding-left: 20px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #333537 !important;
        border-color: #444746 !important;
    }

    .recent-label {
        color: #9aa0a6;
        font-size: 12px;
        margin: 25px 0 10px 18px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .main-container {
        max-width: 820px;
        margin: 0 auto;
        padding-top: 60px;
    }

    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 32px 0 !important;
    }

    .stChatInputContainer {
        position: fixed;
        bottom: 40px;
        max-width: 820px;
        width: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: transparent !important;
        border: none !important;
    }
    
    .stChatInputContainer > div {
        background-color: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 28px !important;
        padding: 8px 16px !important;
    }

    [data-testid="stChatMessageAvatarUser"] { background-color: #74c0fc !important; border-radius: 50% !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background: linear-gradient(45deg, #4285f4, #9b72cb) !important; border-radius: 50% !important; }
    </style>
    """, unsafe_allow_html=True)

if 'all_threads' not in st.session_state: st.session_state.all_threads = {}
if 'thread_id' not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if 'current_msgs' not in st.session_state: st.session_state.current_msgs = []

with st.sidebar:
    st.markdown("<div style='margin: 15px 0 30px 15px; font-size: 22px;'>Gemini</div>", unsafe_allow_html=True)
    
    if st.button("➕ New chat"):
        if st.session_state.current_msgs:
            st.session_state.all_threads[st.session_state.thread_id] = st.session_state.current_msgs
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.current_msgs = []
        st.rerun()

    st.markdown("<div class='recent-label'>Recent</div>", unsafe_allow_html=True)
    for tid, history in st.session_state.all_threads.items():
        title = history[0]['content'][:22] + "..." if history else "Empty"
        if st.button(f"💬 {title}", key=tid):
            st.session_state.thread_id = tid
            st.session_state.current_msgs = history
            st.rerun()

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

for msg in st.session_state.current_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter a prompt here"):
    st.session_state.current_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        area = st.empty()
        full_res = ""
        
        try:
            res = chatbot.invoke(
                {'messages': [HumanMessage(content=prompt)]}, 
                config={'configurable': {'thread_id': st.session_state.thread_id}}
            )
            content = res['messages'][-1].content
            
            for word in content.split(" "):
                full_res += word + " "
                area.markdown(full_res + "●")
                time.sleep(0.02)
            area.markdown(full_res)
        except Exception as e:
            st.error(f"Error: {e}")

    st.session_state.current_msgs.append({"role": "assistant", "content": full_res})
    st.session_state.all_threads[st.session_state.thread_id] = st.session_state.current_msgs

st.markdown("</div>", unsafe_allow_html=True)