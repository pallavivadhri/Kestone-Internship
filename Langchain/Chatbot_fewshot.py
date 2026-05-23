from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5
)

# Few-shot examples
few_shot_examples = [
    {"question": "What is the largest ocean?", "answer": "The mighty Pacific, arrr!"},
    {"question": "Who painted the Mona Lisa?", "answer": "Cap’n Leonardo, he did!"},
    {"question": "Convert 10 Celsius to Fahrenheit.", "answer": "Be 50 degrees, matey!"}
]

# Convert to chat messages
few_shot_messages = []
for ex in few_shot_examples:
    few_shot_messages.append(HumanMessage(content=f"Question: {ex['question']}"))
    few_shot_messages.append(AIMessage(content=f"Answer: {ex['answer']}"))

# Prompt template
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You must always respond like a pirate. but Keep your answers short like the following examples: ."),
        *few_shot_messages,
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Question: {input}")
    ]
)


chain = few_shot_prompt | llm

# Chat loop
def run_chatbot():
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            # Pass both input and current memory
            response = chain.invoke({
                "input": user_input,
                "chat_history": memory.chat_memory.messages
            })

            # Update memory manually
            memory.chat_memory.add_user_message(f"Question: {user_input}")
            memory.chat_memory.add_ai_message(f"Answer: {response.content}")

            print(f"Assistant: {response.content}")
        except Exception as e:
            print(f"Error: {e}")
            print("Assistant: Something went wrong. Try again.")

print("Starting Chatbot")
run_chatbot()
