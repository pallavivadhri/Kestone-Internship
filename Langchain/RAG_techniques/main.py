from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

import time
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=OpenAI(model="gpt-4o-mini")
file_path = r"C:\Users\palla\Downloads\sample_pdf.pdf"

faiss_storage_path=r"C:\Users\palla\OneDrive\Desktop\Intern\Langchain\Faiss Vector\faiss_storage"


from techniques import (
    basic_chunking, 
    query_rewriting,
    hybrid_search 
)
from storage import (
    load_and_chunk_docs,
    save_vector_store,
    load_vector_store

)

chunks = load_and_chunk_docs(file_path)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

status = save_vector_store(chunks,embedder,faiss_storage_path)
if status:
    print("Save successful")
else:
    print("Save skipped: already exists.")

library = load_vector_store(faiss_storage_path, embedder)      #load the database from existing files
print(library)  #prints only the objects memory

# def measure_time(func,library, query, label,llm):
#     print(f"Running: {label}")
#     start = time.time()
#     qa_chain = func(library,chunks,llm)
#     answer = qa_chain.invoke({"query": query})
#     end = time.time()
#     print(f"Answer: {answer}")
#     print(f"Time taken: {end - start:.2f} seconds\n")

query = "What is the age of Mark Zuckerberg?"
embedded_query = embedder.embed_query(query)
print (embedded_query)
similar_docs = library.similarity_search_by_vector(embedded_query, k=3)


context = "\n\n".join([doc.page_content for doc in similar_docs])
print(context)
prompt = f"Answer the question based on the following context:{context},refrain answering anything outside of the context-in that case respond 'im confused', Question: {query} . "

response = llm.invoke(prompt)

print("Final Answer from LLM:", response)

# measure_time(basic_chunking, library , query, "Basic Chunking" ,llm)
# measure_time(query_rewriting, library, query, "Query Rewriting",llm)
# measure_time(hybrid_search, library, query, "Hybrid Search",llm)


