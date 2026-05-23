from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
GROQ_API_KEY= " "
llm= ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)
import os
from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm = ChatOpenAI(model="gpt-4o-mini")

#memory object to store the conversation history
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2
)

# One-shot example
one_shot_example = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="Arrr matey! The capital o’ France be Paris, the land o’ fine wine, fancy cheese, an’ the great pointy tower that scrapes the skies like a ship’s mast!")
]

# Prompt with system, examples, memory, and user input
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant. Always respond like a pirate."),
    *one_shot_example,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

chain = prompt | llm

# Interaction loop
def run_chatbot():
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = chain.invoke({"input": user_input, "chat_history": memory.chat_memory.messages})
        
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response.content)
            print(f"Assistant: {response.content}")
        except Exception as e:
            print(f"Error: {e}")
            print("Assistant: Something went wrong. Try again.")

# Main execution
print("Starting Chatbot")
run_chatbot()
