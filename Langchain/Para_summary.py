from langchain_groq import ChatGroq
GROQ_API_KEY= " "
model= ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)


from langchain_core.prompts import PromptTemplate
summary_prompt_template = PromptTemplate.from_template ("summarize the following passage: {paragraph}")
prompt = summary_prompt_template.invoke ({"paragraph": "The Runnable interface is the foundation for working with LangChain components, and it's implemented across many of them, such as language models, output parsers, retrievers, compiled LangGraph graphs and more.This guide covers the main concepts and methods of the Runnable interface, which allows developers to interact with various LangChain components in a consistent and predictable manner. "})
response = model.invoke(prompt)
print(response.content)



response = model.invoke(" give me a random paragraph story of 5 lines")
random_para = response.content
print(random_para)

response2 = model.invoke(f"summarize this: {random_para}")
print(response2.content)
