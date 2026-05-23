from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

import os, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

# Tavily tool
tavily_tool = TavilySearchResults(max_results=5)

# Wikipedia tool
wiki_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# News API tool
def get_latest_news(query: str) -> str:
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&pageSize=3"
    response = requests.get(url).json()
    articles = response.get("articles")
    if not articles:
        return "No news found."
    return "\n\n".join(f"- {a['title']}\n  {a['url']}" for a in articles)

news_tool = Tool(
    name="NewsAPI_Tool",
    func=get_latest_news,
    description="Useful for getting recent news about a company or topic. Input should be a query like 'Apple' or 'latest AI news'"
)

# Agent1 for gathering information
agent1 = initialize_agent(
    tools=[tavily_tool, wikipedia_tool, news_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Function to scrape website
def get_org_data(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'li'])])
        return text.strip()
    except Exception as e:
        return f"Error fetching organization data: {str(e)}"

query_1 = "What is the latest news about Dell today?"
response1 = agent1.invoke(query_1)

tavily_results = tavily_tool.run("Dell official website")
org_url = tavily_results[0]['url'] #if tavily_results else "https://www.dell.com"

org_data = get_org_data(org_url)

summarizer_chain = load_summarize_chain(llm, chain_type="map_reduce")

def agent2_summarize(raw_data: str) -> str:
    documents = [Document(page_content=raw_data)]
    summary = summarizer_chain.invoke(documents)
    return summary

combined_data = f""" From Tavily/Wiki/News tools: {response1} ,From Organization Website ({org_url}):{org_data} """

final_summary = agent2_summarize(combined_data)

print("\n=== Final Summarized Response ===\n")
print(final_summary)
