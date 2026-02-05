import streamlit as st
from backend import agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command

st.set_page_config(page_title="Ticket Booking Agent", layout="centered")
st.title("🤖 Ticket Booking Agent")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

def get_message_content(message):
    content = message.content
    if isinstance(content, list):
        text_content = ""
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_content += item["text"]
            elif isinstance(item, str):
                text_content += item
        return text_content
    return str(content)

# Display chat history
for message in st.session_state.message_history:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Handle user input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Run the agent with the user's input
        # We need to maintain a thread_id for conversation continuity
        config = {"configurable": {"thread_id": "1"}}
        
        # Stream the response
        try:
            # We use stream to get messages as they come
            response_content = ""
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates"
            ):
                 # Handle updates from the agent node
                if "agent" in event:
                    if "messages" in event["agent"]:
                        last_message = event["agent"]["messages"][-1]
                        response_content = get_message_content(last_message)
                        message_placeholder.markdown(response_content)

            # Check if we are interrupted (waiting for approval)
            state = agent.get_state(config)
            # ... (rest of the file checked for similar usage)
            if state.next and 'tools' in state.next:
                 # It's waiting on a tool call, but specifically check for the interrupt
                 # The interrupt info is usually stored in the state tasks if using interrupts
                 pass # Simple tools don't interrupt unless using `interrupt` function

            # Check for interrupts using the checkpointer state
            # The backend uses `interrupt` inside the tool.
            # When `interrupt` is called, the graph stops.
            # We can check `state.tasks[0].interrupts`
            
            if state.tasks and state.tasks[0].interrupts:
                interrupt_value = state.tasks[0].interrupts[0].value
                # interrupt_value is the dict passed to decision=interrupt(...)
                
                st.info("Approval Required for Ticket Booking")
                st.json(interrupt_value)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve"):
                        # Resume with "yes"
                        resume_command = Command(resume="yes")
                        st.session_state.message_history.append({"role": "assistant", "content": "Booking approved. Processing..."})
                        
                        # Continue streaming after resume
                        resume_response = ""
                        for event in agent.stream(resume_command, config=config, stream_mode="updates"):
                             if "agent" in event: # The tool node usually doesn't output messages directly to the stream in 'updates' mode same way
                                pass
                             if "tools" in event:
                                 # Tool output might be here
                                 pass
                        
                        # After tool execution, the agent node runs again to give final answer
                        # We might need to query state or check final complete response
                        
                        # Simpler approach: Just rerun stream with the command
                        for event in agent.stream(resume_command, config=config, stream_mode="updates"):
                             if "agent" in event and "messages" in event["agent"]:
                                 resume_response = get_message_content(event["agent"]["messages"][-1])
                                 message_placeholder.markdown(resume_response)
                        
                        st.session_state.message_history.append({"role": "assistant", "content": resume_response})
                        st.rerun()

                with col2:
                    if st.button("Reject"):
                        resume_command = Command(resume="no")
                        st.session_state.message_history.append({"role": "assistant", "content": "Booking rejected."})
                        
                        resume_response = ""
                        for event in agent.stream(resume_command, config=config, stream_mode="updates"):
                             if "agent" in event and "messages" in event["agent"]:
                                 resume_response = get_message_content(event["agent"]["messages"][-1])
                                 message_placeholder.markdown(resume_response)
                        
                        st.session_state.message_history.append({"role": "assistant", "content": resume_response})
                        st.rerun()
                
                # If interrupted, we don't fix the final message history yet until resolved
                # Ideally, we should show the "Approval Pending" state.
                # For this simple chat app, we can leave it as "waiting".
                
            else:
                # If not interrupted, save the final response
                 if response_content:
                    st.session_state.message_history.append({"role": "assistant", "content": response_content})

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Check for pending interrupts on page load to show buttons if needed
if not user_input:
    config = {"configurable": {"thread_id": "1"}}
    state = agent.get_state(config)
    if state.tasks and state.tasks[0].interrupts:
        with st.chat_message("assistant"):
            interrupt_value = state.tasks[0].interrupts[0].value
            st.info("Approval Required for Ticket Booking")
            st.json(interrupt_value)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve", key="approve_btn"):
                    resume_command = Command(resume="yes")
                    
                    message_placeholder = st.empty()
                    resume_response = ""
                    for event in agent.stream(resume_command, config=config, stream_mode="updates"):
                            if "agent" in event and "messages" in event["agent"]:
                                resume_response = get_message_content(event["agent"]["messages"][-1])
                                message_placeholder.markdown(resume_response)
                                
                    st.session_state.message_history.append({"role": "assistant", "content": resume_response})
                    st.rerun()

            with col2:
                if st.button("Reject", key="reject_btn"):
                    resume_command = Command(resume="no")
                    
                    message_placeholder = st.empty()
                    resume_response = ""
                    for event in agent.stream(resume_command, config=config, stream_mode="updates"):
                            if "agent" in event and "messages" in event["agent"]:
                                resume_response = get_message_content(event["agent"]["messages"][-1])
                                message_placeholder.markdown(resume_response)

                    st.session_state.message_history.append({"role": "assistant", "content": resume_response})
                    st.rerun()