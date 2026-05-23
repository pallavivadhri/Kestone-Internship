import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# Removed ConversationBufferWindowMemory imports

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Assuming your storage.py handles the FAISS logic
from storage import (
    load_and_chunk_docs,
    save_vector_store,
    load_vector_store
)

load_dotenv()

# Secure your token in .env, but keeping your logic for now
os.environ["HUGGINGFACEHUB_API_TOKEN"] = " "

# 1. Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1, # Lower temperature is better for factual doc querying
    max_new_tokens=512
)

# 2. Paths and Setup
file_path = r"C:\Users\palla\Downloads\ITF Assignment -Team 18.pdf"
faiss_storage_path = r"C:\Users\palla\OneDrive\Desktop\Intern\Langchain\Faiss Vector\faiss_storage"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Process Documents (Only needs to run if index doesn't exist)
chunks = load_and_chunk_docs(file_path)
save_vector_store(chunks, embeddings, faiss_storage_path)
library = load_vector_store(faiss_storage_path, embeddings)

# 4. Refined Prompt (No chat_history placeholder)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a document assistant. Answer the user's question using ONLY the context provided below.\n\n"
        "CONTEXT:\n{context}\n\n"
        "If the answer is not contained within the context, say 'I cannot find the answer in the provided documents.'"
    ),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Create Chain
chain = prompt | llm

def run_doc_bot():
    print("--- Document Query Bot Active ---")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nAsk about the document: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        try:
            # Retrieve relevant chunks
            similar_docs = library.similarity_search(query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in similar_docs])

            # Generate Answer
            response = chain.invoke({
                "context": context_text,
                "input": query
            })

            print(f"\nAssistant: {response}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_doc_bot()