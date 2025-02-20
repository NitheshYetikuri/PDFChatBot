import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import time
import tempfile


llm = ChatOllama(model="llama3.2:latest", temperature=0.1)
embeddings = OllamaEmbeddings(model="llama3.2:latest")
set_llm_cache(InMemoryCache())

def chunking(data):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    return data


def retriever(docs):
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(k=5)
    return retriever

@st.cache_data
def generation(query,_retriever):
    prompt =ChatPromptTemplate.from_template( '''
        You are a question answering Chat Bot. Maintain clear and concise answers.
        Only answer questions if the context is available in the provided documents.
        If the context is not available, respond with "The information is not available in the provided documents."
        Additionally, respond to greetings appropriately.
        Context:{context}

        Question:{question}

        ''')
    chain = (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


uploaded_file = st.file_uploader("Upload PDF here", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    chunks=chunking(documents)
    retriever_store=retriever(chunks)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Enter query")
    if query:
        start_time = time.time()
        st.write(f"Query: {query}")
        response_text = generation(query, retriever_store)
        st.write(response_text)
        end_time = time.time()
        st.write(f"Response time: {end_time - start_time:.2f} seconds")

        # Add the query and response to the chat history
        st.session_state.chat_history.append({"query": query, "response": response_text})

    # Display the chat history
    if st.button("Show History"):
        st.write("Chat History:")
        for chat in st.session_state.chat_history:
            st.write(f"Query: {chat['query']}")
            st.write(f"Response: {chat['response']}")

