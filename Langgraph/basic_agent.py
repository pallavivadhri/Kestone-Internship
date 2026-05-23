# langgraph_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")  

tavily_tool = TavilySearchResults(max_results=1)


agent = initialize_agent(
    tools=[tavily_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# Define LangGraph state
class AgentState(dict): pass

# Define LangGraph nodes
def run_agent(state: AgentState):
    query = state["input"]
    result = agent.invoke({"input": query})
    return AgentState({"input": query, "output": result})

# Build the graph
builder = StateGraph(AgentState)

builder.add_node("agent_node", run_agent)
builder.set_entry_point("agent_node")

# Add a direct edge from the agent node to the end state
# This ensures the graph always finishes after this node.
builder.add_edge("agent_node", END)

# Compile graph
graph = builder.compile()

# Run the agent
if __name__ == "__main__":
    question = "What's the latest news on SpaceX?"
    final_state = graph.invoke(AgentState({"input": question}))
    print(final_state["output"])
