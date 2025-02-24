# PDFChatBot

**PDFChatBot** is a Streamlit-based application that allows users to upload PDF documents and interact with them through a chatbot interface. Utilizing LangChain and FAISS for document retrieval and question answering, PDFChatBot provides concise and context-aware responses based on the content of the uploaded PDFs.

## Features

- **PDF Upload**: Easily upload PDF documents.
- **Document Chunking**: Splits documents into manageable chunks using the CharacterTextSplitter with a chunk size of 500 characters and an overlap of 100 characters.
- **Contextual Retrieval**: Uses FAISS to retrieve relevant document sections.
- **Interactive Chat**: Engage with the chatbot to ask questions about the uploaded PDFs.
- **Caching**: Implements in-memory caching to decrease response time for repeated questions and reduce the cost associated with API calls to the LLM.
- **Embeddings**: Utilizes the OllamaEmbeddings model for generating embeddings.
- **LLM Model**: Uses the ChatOllama model (llama3.2:latest) for generating responses.
- **Chat History**: Maintains a history of the chat interactions for reference.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/PDFChatBot.git
    ```
2. Navigate to the project directory:
    ```bash
    cd PDFChatBot
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
2. Upload a PDF document using the file uploader.
3. Enter your query in the chat input to interact with the document.
