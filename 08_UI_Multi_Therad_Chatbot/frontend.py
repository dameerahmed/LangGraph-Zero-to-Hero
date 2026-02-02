import streamlit as st
from backend import chatbot, generate_thread_id, generate_chat_title
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
import json
import os

THREADS_FILE = "threads.json"

# --- UTILS FUNCTIONS ---
def load_threads():
    if os.path.exists(THREADS_FILE):
        with open(THREADS_FILE, "r") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except:
                return []
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
    st.session_state.message_history = [] 
    st.rerun()

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        if state and "messages" in state.values:
            return state.values["messages"]
    except:
        pass
    return []

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini AI Agent", layout="centered")
st.title("🤖 Gemini Agent")

# --- SESSION STATE ---
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = load_threads()

# --- SIDEBAR ---
st.sidebar.title("🤖 Chat History")
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
for thread in st.session_state.chat_thread[::-1]:
    if thread['title'] not in [None, "New Chat"]:
        if st.sidebar.button(thread['title'], key=thread['id'], use_container_width=True):
            st.session_state.thread_id = thread['id']
            history = load_conversation(thread['id'])
            
            temp_history = []
            for msg in history:
                if isinstance(msg, HumanMessage):
                    temp_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    if not msg.tool_calls:
                        temp_history.append({"role": "assistant", "content": msg.content})
            st.session_state.message_history = temp_history
            st.rerun()

# --- DISPLAY CHAT ---
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT HANDLING ---
if user_input := st.chat_input("Sawal pucho..."):
    add_thread_id(st.session_state.thread_id)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Assistant Response Block ---
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        res_container = {"text": ""}

        def ai_only_stream():
            # stream_mode="messages" returns (chunk, metadata)
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # 1. Lazily create/update status box for tools
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` ...", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` ...", state="running")
                    
                    with status_holder["box"]:
                        st.write(f"✅ Tool `{tool_name}` finished.")

                # 2. Stream assistant tokens only from chat_node
                if metadata.get("langgraph_node") == "chat_node" and isinstance(chunk, AIMessage):
                    content = ""
                    # Handle Gemini's list-based content
                    if isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict) and "text" in item:
                                content += item["text"]
                    else:
                        content = str(chunk.content)

                    if content.strip():
                        # Collapse tool box when AI starts speaking
                        if status_holder["box"] is not None:
                            status_holder["box"].update(label="✅ Tools finished", state="complete", expanded=False)
                        
                        res_container["text"] += content
                        yield content

        # Render stream
        ai_message = st.write_stream(ai_only_stream())

        # Final cleanup for status box if it exists
        if status_holder["box"] is not None:
            status_holder["box"].update(state="complete", expanded=False)

    # Save to history
    if ai_message:
        st.session_state.message_history.append({"role": "assistant", "content": ai_message})

        # Auto Title logic
        for thread in st.session_state.chat_thread:
            if thread['id'] == st.session_state.thread_id:
                if thread['title'] in [None, "New Chat"]:
                    with st.spinner("Naming..."):
                        thread['title'] = generate_chat_title(user_input, ai_message)
                        save_threads(st.session_state.chat_thread)
                    st.rerun()