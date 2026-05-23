import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage

from storage import load_and_chunk_docs, save_vector_store, load_vector_store

load_dotenv()

class AgentB:
    """AgentB uses a PDF as context and maintains conversational memory."""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, pdf_path: str, faiss_path: str):
        self.pdf_path = pdf_path
        self.faiss_path = faiss_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=2
        )

        self._setup_vector_store()
        self.chain = self._setup_chain()

    def _setup_vector_store(self):
        chunks = load_and_chunk_docs(self.pdf_path)
        save_vector_store(chunks, self.embeddings, self.faiss_path)
        self.library = load_vector_store(self.faiss_path, self.embeddings)

    def _setup_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Answer the question based on the following context: {context}. "
                "If the answer is not in the context, reply 'I'm confused'."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return prompt | self.llm

    def invoke(self, query: str) -> str:
        embedded_query = self.embeddings.embed_query(query)
        similar_docs = self.library.similarity_search_by_vector(embedded_query, k=3)

        if similar_docs:
            print("\n Retrieved similar documents:")
            for i, doc in enumerate(similar_docs):
                print(f"--- Document {i + 1} ---\n{doc.page_content[:300]}...\n") 
        else:
            print("No similar documents retrieved. The agent might answer from its own knowledge.\n")

        context = "\n\n".join(doc.page_content for doc in similar_docs)

        response = self.chain.invoke({
            "context": context,
            "input": query,
            "chat_history": self.memory.chat_memory.messages
        })

        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)
        print(f"Memory: {self.memory}")

        print("\n Agent's response based on the context above:")
        print(response.content)

        return response.content

