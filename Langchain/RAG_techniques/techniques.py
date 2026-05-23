from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever



#RAG Techniques
def basic_chunking(library,chunks,llm):
    retriever = library.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def query_rewriting(library,chunks,llm):
   
    base_retriever = library.as_retriever()
    retriever = MultiQueryRetriever.from_llm(llm=llm , retriever=base_retriever)   #rephrase the input query into multiple variations.
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def hybrid_search(library,chunks,llm):
    
    faiss_retriever = library.as_retriever()                #semantic matching
    bm25_retriever = BM25Retriever.from_documents(chunks)   #keyword matching , BM= best matching
    ensemble_retriever = EnsembleRetriever(                 #combines multiple retrievers
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5] # Equal weighting for simplicity
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=ensemble_retriever)
