from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from src.pdf_loader import BytesIOPyMuPDFLoader

def create_embeddings(index, pdf_file, collection):
    
    # creating embeddings
    # load pdf file
    loader = BytesIOPyMuPDFLoader(pdf_file)
    documents = loader.load()
    for doc in documents:
        doc.metadata['file_name'] = pdf_file.name

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Set up embeddings and vector store
    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model=st.secrets["Gemini"]["EMBED_MODEL"])
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        texts, embeddings, collection=collection, index_name="vector_index"
    )

    return vector_store
