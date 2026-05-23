from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent , AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

serpapi_wrapper = SerpAPIWrapper(
    serpapi_api_key= SERPAPI_API_KEY ,
)
search_tool = Tool (
    name= "Search" ,
    func=serpapi_wrapper.run,
    description="Useful for answering questions about current events or recent news."
)
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

response = agent.invoke("What all events happened in india on 12th june 2025 ? Tell me point wise")
print(response)
