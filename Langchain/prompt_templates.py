from langchain_groq import ChatGroq
GROQ_API_KEY= " "
model= ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY)

# prompt templates
# take in raw user input and return data (a prompt) that is ready to pass into the model

#CHATPROMPT TEMPLATE : structure : list of messages with roles
from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"
chatprompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = chatprompt_template.invoke({"language": "Hindi", "text": "how are you"})
print(prompt)
print(model.invoke(prompt))


#PROMPT TEMPLATE : structure : plain string with placeholders
from langchain_core.prompts import PromptTemplate
movie_prompt_template = PromptTemplate.from_template ("tell me a/an {language} movie title of the {genre} genre")
prompt = movie_prompt_template.invoke ({"language":"English" , "genre" :"horror"})
response = model.invoke(prompt)
print(response.content)


#STRINGPROMPT TEMPLATE : structure : define your own .format() logic in a subclass.
from langchain_core.prompts import StringPromptTemplate

class MyPrompt(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        return f"My favorite color is {kwargs['color']}."

prompt = MyPrompt(input_variables=["color"])
print(prompt.format(color="blue"))
response = model.invoke(prompt.format(color="blue"))
print(response)


#FEWSHOTPROMPT TEMPLATE : structure : example input-output pairs to guide the model
from langchain.prompts import FewShotPromptTemplate
examples = [
    {"input": "Hi", "output": "Hola"},
    {"input": "Thanks", "output": "Gracias"},
]
example_prompt = PromptTemplate.from_template("English: {input}\nSpanish: {output}")
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate to Spanish:\n",
    suffix="English: {input}\nSpanish:",
    input_variables=["input"]
)
print(few_shot.format(input="Goodbye"))
response = model.invoke(few_shot.format(input="Goodbye"))
print(response)
