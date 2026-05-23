from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import re

import os
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

tavily_tool = TavilySearchResults(max_results=10)

wiki_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

def add_numbers_from_query(query: str) -> str:
    numbers = re.findall(r"-?\d+", query)
    a, b = map(int, numbers[:2])
    result = a + b
    return str(result)

addition_tool = Tool(
    name="Addition_Tool", 
    func=add_numbers_from_query,
    description="Adds two integers found in the input. Example: 'add 3 and 5' or 'what is 2+2?'"
)

agent = initialize_agent(
    tools=[tavily_tool, wikipedia_tool, addition_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)
# response1 = agent.invoke("what is the latest news today?")
# print(response1)

# response2 = agent.invoke("tell me about virat kohli")
# print(response2)


response1 = agent.invoke("tell me the latest job postings in india for data scientist role, pointwise")
print(response1)