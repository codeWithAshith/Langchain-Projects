import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A Chatbot With Ollama"

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)


def create_vector_embedding(pdf_file):
    if "vectors" not in st.session_state:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, "db")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        docs = PyPDFLoader(pdf_file).load()

        print(f"Number of documents loaded: {len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents=docs)

        print(
            f"Number of final documents: {len(final_documents)}")

        st.session_state.vectors = Chroma.from_documents(documents=final_documents,
                                                         embedding=embeddings,
                                                         persist_directory=persist_directory)


st.title("RAG Document Q&A With Groq And Lama3")

uploaded_file = st.file_uploader("Upload your PDF file")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        create_vector_embedding("temp.pdf")
        st.write("Vector Database is ready")

user_prompt = st.text_input("Enter your query from the recipe document")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
