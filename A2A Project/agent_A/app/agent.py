import os
import re
import requests
from typing import TypedDict, List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain_tavily import TavilySearch

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from firecrawl import FirecrawlApp
from langgraph.graph import StateGraph, END

load_dotenv()


class AgentA:
    class ProjectState(TypedDict):
        input: str
        retrieved_content: str
        scraped_content: str
        final_summary: str
        urls: List[str]
        
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self.tools = self._initialize_tools()
        self.agent2 = self._initialize_agent2()
        self.agent3 = self._initialize_agent3()
        self.graph_executor = self._build_graph()

    def _initialize_tools(self):
        tavily_tool = TavilySearch(max_results=3)

        serpapi_tool = Tool(
            name="Search",
            func=SerpAPIWrapper(
                serpapi_api_key=os.getenv("SERPAPI_API_KEY")
            ).run,
            description="Useful for answering questions about current events or recent news."
        )

        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        news_tool = Tool(
            name="NewsAPI_Tool",
            func=self._get_latest_news,
            description="Get recent news about a company or topic."
        )

        firecrawl_tool = Tool(
            name="FirecrawlScraper",
            func=self._firecrawl_scrape,
            description="Scrapes a single URL using Firecrawl."
        )

        return {
            "tavily": tavily_tool,
            "serpapi": serpapi_tool,
            "wikipedia": wikipedia_tool,
            "news": news_tool,
            "firecrawl": firecrawl_tool
        }

    def _initialize_agent2(self):
        prompt = PromptTemplate(
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

        return initialize_agent(
            tools=[
                self.tools["tavily"],
                self.tools["serpapi"],
                self.tools["news"],
                self.tools["wikipedia"]
            ],
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs={"system_message": prompt.template}
        )

    def _initialize_agent3(self):
        prompt = PromptTemplate(
            input_variables=["input"],
            template=(
                "You are an intelligent webscraper.\n\n"
                "Find all the links in the data given to you, keep the rest of the text as it is and include in the summary.\n"
                "Use the Firecrawl tool to scrape the homepage (e.g., www.companyname.com) and all the links.\n"
                "- Return a detailed summary of the topic.\n\n"
                "User query: {input}"
            )
        )

        return initialize_agent(
            tools=[self.tools["firecrawl"]],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": prompt.template}
        )

    def _build_graph(self):
        builder = StateGraph(self.ProjectState)
        builder.add_node("main agent", self._run_main)
        builder.add_node("retriever", self._run_retriever)
        builder.add_node("scraper", self._run_scraper)
        builder.set_entry_point("main agent")

        builder.add_conditional_edges(
            "main agent", 
            self._should_scrape, {
                "retriever": "retriever",
                "scraper": "scraper",
                "end": END
            }
        )

        builder.add_edge("retriever", "main agent")
        builder.add_edge("scraper", "main agent")

        return builder.compile()

    def _run_main(self, state: ProjectState) -> dict:
        if "retrieved_content" not in state:
            return {"input": state["input"]}

        if "scraped_content" not in state:
            urls = self._extract_urls(state["retrieved_content"])
            if urls:
                return {"urls": urls}
            else:
                combined = f"Retrieved Info:\n{state['retrieved_content']}\n\n(No scraped content)"
                return {
                    "scraped_content": "",
                    "final_summary": self._summarize_text(combined)
                }

        if "final_summary" not in state:
            combined = f"Retrieved Info:\n{state['retrieved_content']}\n\nScraped Content:\n{state['scraped_content']}"
            return {"final_summary": self._summarize_text(combined)}

        return state

    def _run_retriever(self, state: ProjectState) -> dict:
        response = self.agent2.invoke({"input": state["input"]})
        return {"retrieved_content": response["output"]}

    def _run_scraper(self, state: ProjectState) -> dict:
        input_text = "\n".join(state.get("urls", [])) or state["input"]
        result = self.agent3.invoke({"input": input_text})
        return {"scraped_content": result["output"]}

    def _should_scrape(self, state: ProjectState) -> str:
        if state.get("final_summary"):
            return "end"
        if not state.get("retrieved_content"):
            return "retriever"
        if not state.get("scraped_content"):
            return "scraper" if self._extract_urls(state["retrieved_content"]) else "end"
        return "end"

    def _get_latest_news(self, query: str) -> str:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={os.getenv('NEWS_API_KEY')}&language=en&pageSize=3"
        print(f"Query invoked : {query}")
        try:
            response = requests.get(url).json()
            articles = response.get("articles", [])
            if not articles:
                return "No news found."
            return "\n\n".join(f"- {a['title']}\n  {a['url']}" for a in articles)
        except Exception as e:
            return f"Failed to fetch news: {str(e)}"

    def _firecrawl_scrape(self, url: str) -> str:
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        result = app.scrape_url(url, formats=["markdown"])
        return result.markdown[:2000] if result.markdown else "No content returned."

    def _summarize_text(self, content: str) -> str:
        prompt = (
            "You are a smart summarizer. Your job is to convert long content into well-structured summaries.\n\n"
            "Always:\n"
            "- Use bullet points and sub-points.\n"
            "- Focus on clarity, facts, and organization.\n"
            "- Maintain important details but remove redundancy.\n\n"
            f"Content to summarize:\n\n{content}"
        )
        return self.llm.invoke(prompt).content.strip()

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r'https?://\S+', text)

    def invoke(self, query: str) -> dict:
        return self.graph_executor.invoke({"input": query})
    
def llm_search_txt_file(user_query: str, file_path: str = "memory_cache.txt") -> str | None:
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        memory_data = f.read()

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = (
        "You are a smart memory agent.\n"
        "Given the full history of retrieved, scraped, and summarized content from past queries, "
        "find any entries relevant to the current query.\n"
        "Instructions:\n"
        "- Carefully analyze retrieved, scraped, and summary sections.\n"
        "- If multiple entries are relevant, combine insights.\n"
        "- Return a fresh, structured summary relevant to the current query.\n"
        "- If nothing is related, respond with 'No relevant memory found.'\n"
        f"User Query:\n{user_query}\n\n"
        f"Stored Memory Logs:\n{memory_data}"
    )

    response = llm.invoke(prompt).content.strip()
    return None if "no relevant memory found" in response.lower() else response

def save_memory_to_txt(query: str, retrieved: str, scraped: str, summary: str, file_path: str = "memory_cache.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"---\nQUERY: {query}\nRETRIEVED:\n{retrieved}\nSCRAPED:\n{scraped}\nSUMMARY:\n{summary}\n")
