
# using gemma api key
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)

# # GoogleGenerativeAI + PydanticOutputParser + Chain Syntax
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_google_genai import GoogleGenerativeAI
# from pydantic import BaseModel, Field
# import google.generativeai as genai

# # Step 1: API Key Auth (no ADC needed)
# genai.configure(api_key="your_api_key_here")  # Replace with your real API key

# # Step 2: Define the Gemini model
# model = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# # Step 3: Create a Pydantic schema
# class Person(BaseModel):
#     name: str = Field(description="Name of the person")
#     age: int = Field(gt=18, description="Age of the person")
#     city: str = Field(description="City the person belongs to")

# # Step 4: Define parser
# parser = PydanticOutputParser(pydantic_object=Person)

# # Step 5: Prompt template with format instructions
# template = PromptTemplate(
#     template="Generate the name, age and city of a fictional {place} person.\n{format_instruction}",
#     input_variables=["place"],
#     partial_variables={"format_instruction": parser.get_format_instructions()}
# )

# # Step 6: Chain the template, model, and parser
# chain = template | model | parser

# # Step 7: Invoke the chain
# result = chain.invoke({"place": "Sri Lankan"})
# print(result)




# eed to set up Application Default Credentials (ADC)
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel, Field

# load_dotenv()  #  GOOGLE_API_KEY is set in your .env in same directory

# # Define Gemini model
# model = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     temperature=0.7
# )

# # Define the Pydantic schema
# class Person(BaseModel):
#     name: str = Field(description='Name of the person')
#     age: int = Field(gt=18, description='Age of the person')
#     city: str = Field(description='Name of the city the person belongs to')

# # Create parser from schema
# parser = PydanticOutputParser(pydantic_object=Person)

# # Prompt template with format instruction
# template = PromptTemplate(
#     template="Generate the name, age and city of a fictional {place} person.\n{format_instruction}",
#     input_variables=["place"],
#     partial_variables={"format_instruction": parser.get_format_instructions()}
# )

# # Combine everything into a chain
# chain = template | model | parser

# # Run the chain
# final_result = chain.invoke({'place': 'Sri Lankan'})

# print(final_result)



