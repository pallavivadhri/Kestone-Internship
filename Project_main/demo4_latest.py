import os, requests, re
from bs4 import BeautifulSoup
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langgraph.graph import StateGraph, END

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini")

tavily_tool = TavilySearchResults(max_results=5)
serpapi_wrapper = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
serp_tool = Tool(
    name="Search",
    func=serpapi_wrapper.run,
    description="Useful for answering questions about current events or recent news."
)
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

def summarize_text_custom(input: str) -> str:
    prompt = (
        "You are a smart summarizer. Your job is to convert long content into well-structured summaries.\n\n"
        "Always:\n"
        "- Use bullet points and sub-points.\n"
        "- Focus on clarity, facts, and organization.\n"
        "- Maintain important details but remove redundancy.\n\n"
        f"Content to summarize:\n\n{input}"
    )
    return llm.invoke(prompt).content


agent2_prompt_template = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an intelligent research agent with access to multiple tools.\n\n"
        "If the query is about a company, firm, startup, or news, you MUST use ALL 3 tools:\n"
        "- Tavily for web search. \n"
        "- SerpAPI to fetch raw search engine results.\n"
        "- NewsAPI to fetch recent developments.\n\n"
        "If the query is about a general topic, you MUST:\n"
        "- Use Wikipedia.\n"
        "- Return a detailed summary.\n\n"
        "User query: {input}"
    )
)

agent2 = initialize_agent(
    tools=[tavily_tool, news_tool, wikipedia_tool, serp_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": agent2_prompt_template.template}
)


agent3_prompt_template = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an intelligent webscraper.\n\n"
        "find all the links in the data given by you, keep the rest of the text as it is and include in the summary.\n"
        "Use the Webscrape tool to scrape the homepage (e.g., www.companyname.com) and all the links\n"
        "- Return a detailed summary of the topic.\n\n"
        "User query: {input}"
    )
)

agent3 = initialize_agent(
    tools=[scrape_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": agent3_prompt_template.template}
)


class projectstate(TypedDict):
    input: str
    retrieved_content: str
    scraped_content: str
    final_summary: str
    urls: list[str]

def extract_urls(text: str) -> list:
    return re.findall(r'https?://\S+', text)

def run_main(state: projectstate):
    if "retrieved_content" not in state:
        return {"input": state["input"]}

    if "scraped_content" not in state:
        urls = extract_urls(state["retrieved_content"])
        if urls:
            return {"urls": urls}
        else:
            combined = f"Retrieved Info:\n{state['retrieved_content']}\n\n(No scraped content)"
            summary = summarize_text_custom(combined)
            return {"scraped_content": "", "final_summary": summary}

    if "final_summary" not in state:
        combined = f"Retrieved Info:\n{state['retrieved_content']}\n\nScraped Content:\n{state['scraped_content']}"
        summary = summarize_text_custom(combined)
        return {"final_summary": summary}
    return state

def run_retriever(state: projectstate):
    response = agent2.invoke({"input": state["input"]})
    return {"retrieved_content": response["output"]}

def run_scraper(state: projectstate):
    result = agent3.invoke({"input": "\n".join(state["urls"])})
    return {"scraped_content": result["output"]}

def should_scrape(state: projectstate) -> str:
    if "retrieved_content" not in state:
        return "retriever"
    if "scraped_content" not in state and extract_urls(state["retrieved_content"]):
        return "scraper"
    return "end"


builder = StateGraph(projectstate)

builder.add_node("main agent", run_main)
builder.add_node("retriever", run_retriever)
builder.add_node("scraper", run_scraper)

builder.set_entry_point("main agent")

builder.add_conditional_edges(
    "main agent",
    should_scrape,
    {
        "retriever": "retriever",
        "scraper": "scraper",
        "end": END 
    }
)

builder.add_edge("retriever", "main agent")
builder.add_edge("scraper", "main agent")

graph_executor = builder.compile()


from IPython.display import Image, display
try:
    display(Image(graph_executor.get_graph().draw_mermaid_png()))
except Exception:
    pass

with open("graph_latest.png", "wb") as f:
    f.write(graph_executor.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    user_query = input("\nEnter your query: ")
    initial_state = {"input": user_query}
    final_state = graph_executor.invoke(initial_state)
    
    print("\n RETRIVED CONTENT : \n", final_state["retrieved_content"])
    print("\n SCRAPED CONTENT : \n", final_state["scraped_content"])
    print("\n FINAL SUMMARY : \n", final_state["final_summary"])