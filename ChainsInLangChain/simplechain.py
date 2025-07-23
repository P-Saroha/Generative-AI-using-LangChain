## This code is a simple example of using LangChain to create a chain that generates 
# facts about a given topic. using gemini api key
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variable from .env file
load_dotenv()

# Setup Gemini model with API key from environment
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Define the prompt
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Output parser for plain string output
parser = StrOutputParser()

# Build the chain
chain = prompt | model | parser

# Invoke the chain with a topic
result = chain.invoke({'topic': 'cricket'})
print(result)

# Optional: Display the chain structure (if supported)
try:
    chain.get_graph().print_ascii()
except Exception as e:
    print("Graph visualization not supported:", e)





## This code is a simple example of using LangChain to create a chain that generates facts about a given topic.
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# prompt = PromptTemplate(
#     template='Generate 5 interesting facts about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({'topic':'cricket'})

# print(result)

# chain.get_graph().print_ascii()