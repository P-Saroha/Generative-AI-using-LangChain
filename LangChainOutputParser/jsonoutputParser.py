## using gemini api key 
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

# Setup Gemini model
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# JSON parser
parser = JsonOutputParser()

# Prompt with formatting instructions injected
template = PromptTemplate(
    template='Give me 5 facts about {topic}\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Build the chain
chain = template | model | parser

# Invoke the chain
result = chain.invoke({'topic': 'black hole'})

print(result)





# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

# load_dotenv()

# # Define the model
# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# parser = JsonOutputParser()

# template = PromptTemplate(
#     template='Give me 5 facts about {topic} \n {format_instruction}',
#     input_variables=['topic'],
#     partial_variables={'format_instruction': parser.get_format_instructions()}
# )

# chain = template | model | parser

# result = chain.invoke({'topic':'black hole'})

# print(result)
