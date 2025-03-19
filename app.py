import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Check if API key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 OpenAI API key not found! Please check your .env file and restart the app.")
    st.stop()  # Stop execution if API key is missing

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Function to handle user queries
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main Streamlit app function
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    
    # Inject CSS and FontAwesome for icons
    st.write("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """ + css, unsafe_allow_html=True)

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # App Title
    st.header("📚 Chat with Your PDFs")
    
    # User Input for Querying PDFs
    user_question = st.text_input("🔍 Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Sidebar for PDF Upload
    with st.sidebar:
        st.subheader("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("🚀 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing documents... "):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text
                    text_chunks = get_text_chunks(raw_text)  # Split text
                    vectorstore = get_vectorstore(text_chunks)  # Create vector store
                    st.session_state.conversation = get_conversation_chain(vectorstore)  # Setup chat
                    st.session_state.chat_history = None  # Reset chat history
                st.success(" Documents processed successfully! You can start asking questions.")
            else:
                st.error(" No PDFs uploaded! Please upload at least one file.")

if __name__ == '__main__':
    main()
