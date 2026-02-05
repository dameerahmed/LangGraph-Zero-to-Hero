import streamlit as st
from backend_chatbot import chatbot
from langchain_core.messages import HumanMessage, BaseMessage
st.set_page_config(page_title="LangGraph Chatbot", layout="centered")
st.title("🤖 My Agentic AI")

# 1. Session State Initialize
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# 2. Purani Chat History Display karna
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. User Input Handle karna
if user_input := st.chat_input("Sawal pucho..."):
    
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
                config={"configurable": {"thread_id": "1"}},
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

        # write_stream automatic typewriter effect deta hai
        try:
            ai_message = st.write_stream(response_generator())
            # Final response ko history mein save karna
            st.session_state.message_history.append({"role": "assistant", "content": ai_message})
        except Exception as e:
            st.error(f"Streaming mein masla aaya: {e}")