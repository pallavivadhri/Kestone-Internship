from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini") 

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True, # If False:returns plain strings. If True: full BaseMessage objects like HumanMessage, AIMessage
    k=5 # Keep last 5 messages in memory
)

SYSTEM_PROMPT = """
You are a helpful and friendly AI assistant.You can answer questions, provide information, and engage in casual conversation.
"""

# Base Chat Prompt Template with Memory Placeholder
base_chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"), # This is where memory is inserted
        HumanMessagePromptTemplate.from_template("{input}"), # User's current query
    ]
)

# One-Shot Prompt (Example of specific behavior)
def get_one_shot_response(query: str):
    one_shot_example = [
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris.")
    ]
    
    # Combine system, one-shot example, and user query
    one_shot_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful assistant. Always respond concisely."),
            *one_shot_example, 
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    
    chain = one_shot_prompt_template | llm
    response = chain.invoke({"input": query})
    return response.content

# Few-Shot Prompt (Examples of multiple desired behaviors)
def get_few_shot_response(query: str):
    few_shot_examples = [
        {"question": "What is the largest ocean?", "answer": "Pacific Ocean."},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci."},
        {"question": "Convert 10 Celsius to Fahrenheit.", "answer": "50 Fahrenheit."}
    ]

    few_shot_messages = []
    for ex in few_shot_examples:
        few_shot_messages.append(HumanMessage(content=f"Question: {ex['question']}"))
        few_shot_messages.append(AIMessage(content=f"Answer: {ex['answer']}"))

    few_shot_chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("You are a helpful assistant. Here are some examples of how to answer questions:"),
            *few_shot_messages,
            HumanMessagePromptTemplate.from_template("Question: {input}")
        ]
    )
    chain = few_shot_chat_template | llm
    response = chain.invoke({"input": query})
    return response.content

# Interaction Loop
def run_chatbot():
    print("Hi, i am here to help you today. Ask me anything")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("bye")
            break

        try:
            # current chat history from memory
            chat_history = memory.load_memory_variables({})["chat_history"]

            # Invoke the chain with the current input and chat history
            
            chain = base_chat_prompt | llm
            result = chain.invoke({"input": user_query, "chat_history": chat_history})
            
            # Update memory with the new interaction
            memory.save_context({"input": user_query}, {"output": result.content})

            print(f"{result.content}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("I might have faced an issue processing that. Please try rephrasing.")

# Main execution
if __name__ == "__main__":
    print("Starting Chatbot")
    run_chatbot()

    print("\n Demonstrating One-Shot Prompt")
    one_shot_result = get_one_shot_response("What is the highest mountain?")
    print(f"One-shot response: {one_shot_result}")

    print("\n Demonstrating Few-Shot Prompt")
    few_shot_result = get_few_shot_response("What is the chemical symbol for water?")
    print(f"Few-shot response: {few_shot_result}")