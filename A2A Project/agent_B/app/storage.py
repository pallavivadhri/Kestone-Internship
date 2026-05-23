import os
import pickle
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def load_and_chunk_docs(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print("Document loaded successfully.")

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Document split into {len(chunks)} chunks.")
    return chunks


def save_vector_store(chunks, embedder, faiss_storage_path):
    if os.path.exists(faiss_storage_path):
        return False  

    vectorstore = FAISS.from_documents(chunks, embedder)
    print("Document is embedded")
    vectorstore.save_local(faiss_storage_path)
    return True


def load_vector_store(faiss_storage_path , embedder):
    
    vectorstore = FAISS.load_local(faiss_storage_path, embedder,allow_dangerous_deserialization=True)
    print(f"Loaded vector store from '{faiss_storage_path}'")
    return vectorstore