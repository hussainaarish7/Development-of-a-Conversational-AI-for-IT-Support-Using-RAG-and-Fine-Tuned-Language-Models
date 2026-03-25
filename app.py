import streamlit as st
from utils import create_qa_chain

st.set_page_config(page_title="IT Support Chatbot", layout="centered")

st.title("💻 IT Support Chatbot")
st.write("Ask any IT-related question!")

qa = create_qa_chain()

query = st.text_input("Enter your question:")

if query:
    response = qa.run(query)
    st.success(response)
