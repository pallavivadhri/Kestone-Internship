from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict

import os
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")


prompt = hub.pull("hwchase17/react")
tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

# Agent 1: Researcher 
agent1 = create_react_agent(llm, tools, prompt)
agent1_executor = AgentExecutor(agent=agent1, tools=tools, verbose=True)
print("Agent 1 (Researcher) initialized")


def summarize_text(input: str) -> str:
    response = llm.invoke(f"Summarize the following content briefly:\n\n{input}")
    return response.content

summarize_tool = Tool.from_function(
    name="SummarizeText",
    description="Summarizes long text into concise paragraphs",
    func=summarize_text
)

agent2 = initialize_agent(
    tools=[summarize_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
print("Agent 2 (Summarizer) initialized")


class twoagentstate(TypedDict):
    input: str
    agent1_output: str
    agent2_output: str

def run_agent1(state: twoagentstate):
    query = state["input"]
    result = agent1_executor.invoke({"input": query})
    return {"input": query, "agent1_output": result["output"]}


def run_agent2(state: twoagentstate):
    paragraph = state["agent1_output"]
    result = agent2.invoke({"input": paragraph})
    return {"input": state["input"], "agent1_output": paragraph, "agent2_output": result["output"]}

# Build LangGraph
builder = StateGraph(twoagentstate)
builder.add_node("agent1_node", run_agent1)
builder.set_entry_point("agent1_node")
builder.add_node("agent2_node", run_agent2)
builder.add_edge("agent1_node", "agent2_node")
builder.add_edge("agent2_node", END)

graph = builder.compile()

query = "Latest news about India-Pakistan relations in 2025"
output = graph.invoke({"input": query})


print("\n--- Final Outputs ---")
print("Agent 1 Output :", output["agent1_output"])
print("Agent 2 Output :", output["agent2_output"])
