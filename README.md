# PDF Chat Application

A Streamlit-based application that allows you to chat with your PDF documents using OpenAI's GPT models and LangChain for document processing.

## Features

- Upload multiple PDF files
- Extract text from PDFs and split into manageable chunks
- Create vector embeddings using OpenAI's API
- Chat with your documents using conversational AI
- Maintain conversation history during the session

## Setup Instructions

### Phase 1: Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the App**
   - Open your browser and go to `http://localhost:8501`
   - Enter your OpenAI API key
   - Upload PDF files and start chatting!

### Phase 2: Deploy to Streamlit Community Cloud

1. **Create GitHub Repository**
   - Create a new public repository on GitHub
   - Upload `app.py` and `requirements.txt` to the repository

2. **Deploy to Streamlit**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign up with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy!"

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls

## Usage

1. Enter your OpenAI API key in the password field
2. Upload one or more PDF files using the sidebar
3. Click "Process" to analyze your documents
4. Ask questions about your PDF content in the chat interface

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Document processing and conversational AI
- **OpenAI**: GPT models and embeddings
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF text extraction

## Notes

- Your OpenAI API key is stored only in your browser session
- The application processes documents locally before sending queries to OpenAI
- Conversation history is maintained during your session
