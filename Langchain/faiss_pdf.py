from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
file_path = r"C:\Users\palla\Downloads\sample_pdf.pdf"

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path=r"C:\Users\palla\Downloads\sample_pdf.pdf")    
    doc = loader.load()
    return doc

def chunkembedding(doc):
    doc_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 800,
        chunk_overlap=50,
    )
    doc_split = doc_splitter.split_documents(doc)
    embedding = OpenAIEmbeddings(model= "text-embedding-ada-002")
    library = FAISS.from_documents(doc_split,embedding)  #create a vector database
    return library


doc = pdf_loader(file_path)
library = chunkembedding(doc)

#only prints chunks
query = 'Tell me masterpieces of the author'
query_ans = library.similarity_search (query)
print ( query_ans[0].page_content)  #first similar chunk

#answers based on retrieved knowledge
retriever = library.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model= "gpt-4o-mini"), 
    retriever=retriever
)
print(qa_chain.run("Tell me the summary of the text"))
