import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("Loaded API Key:", OPENAI_API_KEY)

from langchain_openai import ChatOpenAI
model= ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)
response = model.invoke("Suggest me a mystery novel to read")
print(response)
print(response.content)