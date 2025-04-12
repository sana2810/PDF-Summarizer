

import os
from config import OPENAI_API_KEY
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document  # <-- Import Document
from PIL import Image
from langchain.chains.question_answering import load_qa_chain

import tempfile
import time

# --- Logo ---
st.image("AI.png", width=150)

# --- Title ---
st.title("📑 AI PDF Summarizer")
st.markdown("Upload a PDF, and look how AI does the magic and summarize it in seconds!!")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("🤖 Loading and summarizing..."):
        # Save file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        # Load model
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        # Progress bar
        progress = st.progress(0)

        # Update progress: Start summarizing
        progress.progress(20)

        # Summarizing with LangChain
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)

        # Update progress: Finished summarizing
        progress.progress(100)

        st.success("✅ Summary generated!")

        # Editable summary
        st.subheader("📝️ Edit Your Summary Below:")
        edited_summary = st.text_area("You can edit the summary if you'd like:", summary, height=300)

        # Ask a question
        st.subheader("🤔 Any questions about the summary")
        user_question = st.text_input("Type your question here:")

        if user_question:
            with st.spinner("🤖 Thinking..."):
                # Use the Document class to create a document for the QA chain
                doc = Document(page_content=edited_summary)
                qa_chain = load_qa_chain(llm, chain_type="stuff")
                response = qa_chain.run(input_documents=[doc], question=user_question)
                st.success(response)

        # Download button
        st.download_button("📥 Download Summary", edited_summary, file_name="summary.txt")


st.markdown("""
        <style>
        .stApp {
            background-color: #f2f7ff;
            color: black;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1f4e79;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stDownloadButton > button {
            background-color: #007bff;
            color: white;
        }
        .stFileUploader {
            background-color: #ffffff;
            border: 2px dashed #007bff;
        }
        .stProgress > div > div {
            background-image: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }
        </style>
    """, unsafe_allow_html=True)
