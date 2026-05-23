from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini")

# Tavily search tool
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
    description="Get recent news about a company or topic. Input should be a company/topic name."
)

# news_data = get_latest_news(query= "Dell")
# print(news_data)

# Website scraper tool
def get_org_data(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', class_='article-content') or soup
        text = ' '.join([p.get_text() for p in main_content.find_all(['p', 'h1', 'h2'])])
        return text.strip()
    except Exception as e:
        return f"Error fetching organization data: {str(e)}"

scrape_tool = Tool(
    name="WebsiteScraper",
    func=get_org_data,
    description="Scrapes website content given a URL."
)

# agent_prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template=(
#         "You are an intelligent research agent with access to multiple tools.\n\n"
#         "If the query is about a company, firm, startup, or news, you MUST:\n"
#         "- Use Tavily for web search. use Webscrape tool and give detailed information about the company website"
#         "- Use NewsAPI to fetch recent developments.\n"
#         "- Return a very detailed explanation using both sources.\n\n"
#         "If the query is about a general topic, you MUST:\n"
#         "- Use Wikipedia.\n"
#         "- Return a detailed summary of the topic.\n\n"
#         "Always choose tools based on the query's nature. Provide full, useful information.\n\n"
#         "User query: {query}\n"
#     )
# )

agent_1 = initialize_agent(
    tools=[tavily_tool, news_tool, wikipedia_tool, scrape_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

detailed_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in very detailed bullet points.\n"
        "Focus on all important facts. Use subpoints where helpful.\n"
        "Text:\n{text}\n\n"
        "Summary:\n"
    )
)
summarizer_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=detailed_summary_prompt,
    combine_prompt=detailed_summary_prompt
)


def summarizer_agent(text: str) -> str:
    documents = [Document(page_content=text)]
    result = summarizer_chain.invoke(documents)
    return result["output_text"]

# Main 
def main_agent(query: str):
    company_keywords = ["company", "organization", "firm", "startup", "latest", "news"]
    is_company_query = any(word in query.lower() for word in company_keywords) or "news about" in query.lower()
    
    if is_company_query:
        print("Detected as company-related query.\n")
        agent_response = agent_1.invoke(query)
        print(f"\nAGENT RESPONSE:\n{agent_response}\n")

        summary = summarizer_agent(agent_response["output"])
        return summary

    else:
        print("Detected as general knowledge query.\n")
        agent_response = agent_1.invoke(query)
        print(f"\nAGENT RESPONSE:\n{agent_response}\n")

        summary = summarizer_agent(agent_response["output"])
        return summary



if __name__ == "__main__":
    user_query = input("Enter your query: ")
    final_response = main_agent(user_query)
    print("\n FINAL RESPONSE (SUMMARY) : \n")
    print(final_response)
