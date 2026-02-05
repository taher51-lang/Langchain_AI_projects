from langgraph.graph import StateGraph, END, START
# 1. Added AIMessage and ToolMessage to imports for filtering
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import streamlit as st
from langgraphbackend import chatbot

def generate_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def get_messages(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    if 'messages' in state.values:
        return state.values['messages']
    return []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_id()
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

st.sidebar.title("Chatbot")

if st.sidebar.button("New chat"):
    reset_chat()

st.sidebar.header("My conversations")
chatName = 0

for thread_id in st.session_state['chat_threads']:
    chatName += 1
    if st.sidebar.button(str(f"Serial No {chatName} {str(thread_id)[0]}")):
        st.session_state['thread_id'] = thread_id
        messages = get_messages(thread_id)
        msg = []
        for message in messages:
            if isinstance(message, ToolMessage):
                continue 
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            msg.append({'role': role, 'content': message.content})
        st.session_state['message_history'] = msg

# Render the chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    config = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
    
        def stream_filtered_response():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            ):
           
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(stream_filtered_response())
    
    st.session_state["message_history"].append({'role': 'assistant', 'content': ai_message})