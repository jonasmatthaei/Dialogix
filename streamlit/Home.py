import sys
import streamlit as st
from datetime import datetime
from streamlit import experimental_rerun
import os
from main import initialize, initial_query_handler, follow_up_handler, setup
from haystack import Pipeline

# APP title
st.set_page_config(page_title="Custom Chat")

# Initialize or retrieve conversations and file upload status
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.file_uploaded = False
    st.session_state.upload_key = 0  # Key to reset file uploader

# Function to create or switch conversation
def manage_conversation(conversation_id=None):
    if conversation_id:
        st.session_state.current_conversation = conversation_id
    else:
        current_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        st.session_state.current_conversation = current_time
        st.session_state.conversations[current_time] = [{"role": "assistant", "content": "Thank you for uploading the file, let's start exploring."}]
    #st.session_state.file_uploaded = False  # Reset file upload status
    #st.session_state.upload_key += 1  # Increment key to reset file uploader

if "current_conversation" not in st.session_state:
    manage_conversation()

# File uploader in sidebar
st.sidebar.title("File Upload")

if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False

if 'uploaded_file_path' not in st.session_state:
    st.session_state['uploaded_file_path'] = ""

uploaded_file = st.sidebar.file_uploader("Upload a file", type=['txt', 'out'], key=st.session_state.upload_key)
if uploaded_file and not st.session_state.file_processed:
    filename = uploaded_file.name

    # Save the file
    with open('./data/' + filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Store the file path in session state for later deletion
    st.session_state['uploaded_file_path'] = filename

    initialize(filename)

    st.session_state.file_processed = True

    st.sidebar.write("File uploaded successfully!")
    st.session_state.file_uploaded = True

elif not uploaded_file:
    st.session_state.file_uploaded = False
    st.session_state.file_processed = False

if st.session_state.get('file_processed', False):
    if not 'document_store' in st.session_state:
        st.session_state['document_store'], st.session_state['retriever'] = setup(st.session_state['uploaded_file_path'])
    document_store = st.session_state['document_store']
    retriever = st.session_state['retriever']

    # Sidebar for conversation history
    st.sidebar.title("Conversation History")
    if st.sidebar.button("Start New Conversation"):
        manage_conversation()

    # List conversations in sidebar
    conversation_keys = sorted(st.session_state.conversations.keys(), reverse=True)
    for key in conversation_keys:
        conversation_title = key
        first_user_message = next((msg["content"] for msg in st.session_state.conversations[key] if msg["role"] == "user"), None)
        if first_user_message:
            conversation_title += f" - {first_user_message[:30]}"
        if st.sidebar.button(conversation_title):
            manage_conversation(conversation_id=key)

    # Display chat messages for the current conversation
    if st.session_state.current_conversation:
        for message in st.session_state.conversations[st.session_state.current_conversation]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    response = None

    # User-provided prompt
    if 'first_query' not in st.session_state:
        st.session_state['first_query'] = True
    st.session_state.firstquery = False
    if st.session_state.current_conversation:
        prompt = st.chat_input("Enter your message:")
        if prompt:
            if not st.session_state.file_uploaded:
                # If file is not uploaded, show a humorous message
                st.warning("Hold your horses! Please upload a log file first. I can't work with thin air!")
            else:
                st.session_state.conversations[st.session_state.current_conversation].append({"role": "user", "content": prompt})
                if st.session_state['first_query']:
                    # Handle initial query
                    response, pipeline, context = initial_query_handler(prompt, retriever, document_store, st.session_state['uploaded_file_path'])
                    if 'pipeline' not in st.session_state:
                        st.session_state['pipeline'] = pipeline
                    st.session_state.context = context
                    st.session_state['first_query'] = False
                else:
                    # Handle follow-up query
                    response = follow_up_handler(prompt, st.session_state['pipeline'], st.session_state.context)

                    if not response:
                        sys.exit()
                    if len(response[0]) < 10:
                        response, pipeline, context = initial_query_handler(prompt, retriever, document_store, st.session_state['uploaded_file_path'])
                        if 'pipeline' not in st.session_state:
                            st.session_state['pipeline'] = pipeline
                        st.session_state.context = context
                        st.session_state['first_query'] = False


                additional_text = "Use prompt 'full logs' to see the complete log files corresponding to the requested events."

                full_response = response[0] + "\n\n"
                if prompt != 'full logs':
                    full_response += additional_text

                message = {"role": "assistant", "content": full_response}
                st.session_state.conversations[st.session_state.current_conversation].append(message)
                experimental_rerun()
