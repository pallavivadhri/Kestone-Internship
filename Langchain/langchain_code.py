
# import a chat model integration
from langchain_openai import ChatOpenAI
OPENAI_API_KEY=" "
model= ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)

# invoke is method to send a prompt
response = model.invoke("Suggest me a mystery novel to read")
print(response)
print(response.content)
 
# another way to give the invoke - use messages
from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("translate the following from english to italian"),
    HumanMessage("hi")
]
response= model.invoke(messages)
print(response.content)
