from langchain_groq import ChatGroq
GROQ_API_KEY= " "
model= ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)

# invoke-method to send a prompt
# langchain automatically converts the input string into HumanMessage object
response = model.invoke("Tell me the top 5 mystery novels of all time")
print(response)
print(response.content)


# another way for input - use messages
from langchain_core.messages import HumanMessage, SystemMessage
print(model.invoke([HumanMessage(content="Hello, how are you?")]))

messages = [
    SystemMessage("You are a translator"), #persona/intructions for the model
    HumanMessage("Translate 'hi' to Italian") #input for the model
]
response= model.invoke(messages)
print(response.content)


ai_message = model.invoke([HumanMessage("tell me a joke ")]) #response of the model
print(ai_message)

#model streams the output as chunks (single words)
response =" "
for chunk in model.stream([HumanMessage("what color is the sky during the day?")]):
    print(chunk)
    response=response + chunk.content
print(response)

