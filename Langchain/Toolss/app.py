from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

tavily_tool = TavilySearchResults(max_results=5)

wiki_api_wrapper = WikipediaAPIWrapper()

wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

agent = initialize_agent(
    tools=[tavily_tool, wikipedia_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)



from flask import Flask
from flask import Flask, request, jsonify
app = Flask(__name__)

#defines the root URL
# @app.route('/')
# def home():
#     return "Hello from local Flask server!"


@app.route('/', methods=['POST'])
def home():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = agent.invoke(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True,port=5000)