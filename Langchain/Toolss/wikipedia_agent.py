from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize the Wikipedia API wrapper
wiki_api_wrapper = WikipediaAPIWrapper()

# Create the Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

result = wikipedia.run(" HOMO SAPIENS")
print(result)


llm = ChatOpenAI(model="gpt-4o-mini")

# Agent with the Wikipedia tool
agent = initialize_agent(
    tools=[wikipedia],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

response = agent.invoke("Tell me briefly about the virat kohli")
print(response)
