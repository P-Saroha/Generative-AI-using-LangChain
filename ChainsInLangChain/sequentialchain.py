# This code demonstrates how to create a sequential chain using LangChain with Google Gemini model.
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

# Setup Gemini model
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Step 1 prompt: Detailed report
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Step 2 prompt: Summarize the text into 5 points
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n{text}',
    input_variables=['text']
)

# Output parser for plain text
parser = StrOutputParser()

# Define the chain (report → summary)
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain
result = chain.invoke({'topic': 'Unemployment in India'})
print(result)

# Optional: Show the chain structure
chain.get_graph().print_ascii()




## This code is part of the LangChain framework, which allows for the
#  creation of complex chains of operations. with the help of openai 

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Generate a detailed report on {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Generate a 5 pointer summary from the following text \n {text}',
#     input_variables=['text']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# chain = prompt1 | model | parser | prompt2 | model | parser

# result = chain.invoke({'topic': 'Unemployment in India'})

# print(result)

# chain.get_graph().print_ascii()