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

# Check if API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error(" OpenAI API key not found! Please check your .env file.")

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Prevent NoneType error
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    """Split extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Create vector store using FAISS and OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Handle user queries and display responses."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        st.write(
            (user_template if i % 2 == 0 else bot_template).replace("{{MSG}}", message.content), 
            unsafe_allow_html=True
        )

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📚")
    
    # User input for querying the PDF
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text
                    text_chunks = get_text_chunks(raw_text)  # Split into chunks
                    vectorstore = get_vectorstore(text_chunks)  # Create vector store
                    st.session_state.conversation = get_conversation_chain(vectorstore)  # Create chat chain
                    st.session_state.chat_history = None  # Reset chat history
                st.success(" Documents processed successfully!")
            else:
                st.error(" No PDFs uploaded! Please upload files first.")

if __name__ == '__main__':
    main()
