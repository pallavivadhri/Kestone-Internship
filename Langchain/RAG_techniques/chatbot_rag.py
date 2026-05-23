from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o-mini")
file_path = r"C:\Users\palla\Downloads\sample_pdf.pdf"

faiss_storage_path=r"C:\Users\palla\OneDrive\Desktop\Intern\Langchain\Faiss Vector\faiss_storage"

from storage import (
    load_and_chunk_docs,
    save_vector_store,
    load_vector_store

)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2
)

chunks = load_and_chunk_docs(file_path)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
save_vector_store(chunks,embeddings,faiss_storage_path)
library = load_vector_store(faiss_storage_path, embeddings)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Answer the question based on the following context:{context}. Refrain answering anything outside of the context-in that case respond 'I'm confused' . "),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

chain = prompt | llm

def run_chatbot():
    print("Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            
            embedded_query = embeddings.embed_query(query)
            #print(embedded_query)
            similar_docs = library.similarity_search_by_vector(embedded_query, k=3)
            context = "\n\n".join([doc.page_content for doc in similar_docs])
            #print(context)
            
            response = chain.invoke({"context":context ,"input": query, "chat_history": memory.chat_memory.messages})
            
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response.content)

            print(f"Assistant: {response.content}")
            
            print(memory)
        except Exception as e:
            print(f"Error: {e}")
            print("Assistant: Something went wrong. Try again.")

if __name__ == "__main__":
    print("Starting Chatbot")
    run_chatbot()


def run_chatbot2(query):
    embedded_query = embeddings.embed_query(query)
    similar_docs = library.similarity_search_by_vector(embedded_query, k=3)
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    response = chain.invoke({
        "context": context,
        "input": query,
        "chat_history": memory.chat_memory.messages
    })

    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response.content)

    return response.content