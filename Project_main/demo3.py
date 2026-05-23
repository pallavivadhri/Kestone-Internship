from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate


import os, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# Load API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# LLM initialization
llm = ChatOpenAI(model="gpt-4o")

# Tools
tavily_tool = TavilySearchResults(max_results=5)

wiki_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

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

# Custom system prompt
agent_prompt_template = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an intelligent research agent with access to multiple tools.\n\n"
        "If the query is about a company, firm, startup, or news, you MUST:\n"
        "- Use Tavily for web search. Use the Webscrape tool to scrape the homepage (e.g., www.companyname.com), not internal links like press releases.\n"
        "- Use NewsAPI to fetch recent developments.\n"
        "- Return a very detailed explanation using both sources.\n\n"
        "If the query is about a general topic, you MUST:\n"
        "- Use Wikipedia.\n"
        "- Return a detailed summary of the topic.\n\n"
        "Always choose tools based on the query's nature. Provide full, useful information.\n\n"
        "User query: {input}"
    )
)

# Agent initialization
agent_1 = initialize_agent(
    tools=[tavily_tool, news_tool, wikipedia_tool, scrape_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": agent_prompt_template.template}
)

# Summarizer prompt and chain
detailed_summary_prompt = PromptTemplate(
    input_variables=["input"],
    template=(
        "Summarize the following text in very detailed bullet points.\n"
        "Focus on all important facts. Use subpoints where helpful.\n"
        "Text:\n{text}\n\n"
        "Summary:\n"
    )
)

summarizer_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=detailed_summary_prompt,
    combine_prompt=detailed_summary_prompt,
    
)

def summarizer_agent(text: str) -> str:
    documents = [Document(page_content=text)]
    result = summarizer_chain.invoke(documents)
    return result["output_text"]

# Main query function
def main_agent(query: str):
    print("Running agent with custom system prompt...\n")

    agent_response = agent_1.run(query)  

    print(f"\nAGENT RAW OUTPUT:\n{agent_response[:1000]}...\n")  

    summary = summarizer_agent(agent_response)
    return summary


# Entry point
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    final_response = main_agent(user_query)
    print("\nFINAL RESPONSE (SUMMARY):\n")
    print(final_response)
