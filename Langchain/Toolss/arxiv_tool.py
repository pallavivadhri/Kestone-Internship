from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import AgentExecutor,create_react_agent ,load_tools
from langchain_openai import ChatOpenAI
from langchain import hub

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o-mini")

print(f"----------USING AGENT ------------")

tools = load_tools(
    ["arxiv"],
)
prompt = hub.pull("hwchase17/react")   #prompt template from the LangChain Hub: REason+ACT
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What's the paper 1605.08386 about?",
    }
)

print(f"---------USING API CALL------------")
arxiv_wrapper = ArxivAPIWrapper()
docs = arxiv_wrapper.run("1605.08386")
print(docs)