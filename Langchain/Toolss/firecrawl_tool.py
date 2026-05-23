import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from firecrawl import FirecrawlApp, ScrapeOptions


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

tavily_tool = TavilySearchResults(max_results=5)

def firecrawl_search(query: str) -> str:
    result = firecrawl_app.search(query, limit=2)
    return "\n".join([r['url'] for r in result.data])

def firecrawl_scrape(url: str) -> str:
    result = firecrawl_app.scrape_url(url, formats=["markdown"])
    content = result.markdown or "No content returned."
    return content[:2000]  

def firecrawl_crawl(url: str) -> str:
    crawl_result = firecrawl_app.crawl_url(
        url, 
        limit=2, 
        scrape_options=ScrapeOptions(formats=['markdown']),
    )
    return crawl_result

def firecrawl_map(url: str) -> str:
    result = firecrawl_app.map_url(url,limit=10)
    return result

tools = [
    #Tool(name="TavilySearch", func=tavily_tool.run, description="Useful for web searches"),
    Tool(name="FirecrawlScraper", func=firecrawl_scrape, description="scrapes only the specified url without crawling subpages. Outputs the content from the page..Input is a valid URL."),
    Tool(name="FirecrawlCrawler", func=firecrawl_crawl, description="crawls a url and all its accessible subpages, outputting the content from each page. Input is a base URL."),
    Tool(name="FirecrawlMapper", func=firecrawl_map, description="attempts to output all website’s urls in a few seconds. Input is a base URL."),
    Tool(name="FirecrawlSearcher", func=firecrawl_search, description="search the web and get full content from results. Input is a search query.")
]

system_prompt = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an intelligent web research agent with access to Firecrawl tools.\n\n"
        "- Use FirecrawlSearcher to find relevant pages.\n"
        "- Then use FirecrawlMapper first. then use FirecrawlScraper AND then FirecrawlCrawler for every link. \n"
        "- Combine all tools to provide a detailed and accurate response.\n\n"
        "Query: {input}"
    )
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt.template}
)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response = agent.invoke(user_query)
    print("\nAGENT RESPONSE\n")
    print(response)
