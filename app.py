import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
# CHANGED: Import Google's embeddings and chat models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    """Creates a FAISS vector store from text chunks using Google embeddings."""
    # CHANGED: Use GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """Creates a conversational retrieval chain."""
    # CHANGED: Use ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=api_key, temperature=0.7
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs (using Gemini) :books:")
    
    # CHANGED: Ask for Google API Key
    google_api_key = st.text_input("Enter your Google API Key:", type="password")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            if not google_api_key:
                st.warning("Please enter your Google API Key first.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Get PDF text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks, google_api_key)
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore, google_api_key
                        )
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        if "quota" in str(e).lower() or "api" in str(e).lower():
                            st.info("💡 **Solutions:**\n1. Check your Google AI Studio API key\n2. Verify your API key has proper permissions\n3. Check your API usage limits")

    # Handle user input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({'question': user_question})
                    st.session_state.chat_history = response['chat_history']

                    # Display chat history
                    for i, message in enumerate(st.session_state.chat_history):
                        if i % 2 == 0:
                            st.write(f"**You:** {message.content}")
                        else:
                            st.write(f"**Bot:** {message.content}")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if "quota" in str(e).lower() or "api" in str(e).lower():
                        st.info("💡 Your Google API quota may have been exceeded. Please check your API usage.")
        else:
            st.warning("Please upload and process your PDFs first.")

if __name__ == '__main__':
    main()
