from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import os

# Load API key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Prompt for tweet
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

# Prompt for LinkedIn post
prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"]
)

# Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Output parser
parser = StrOutputParser()

# Run in parallel
parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser)
})

# Execute chain
result = parallel_chain.invoke({"topic": "AI"})

print(result["tweet"])
print(result["linkedin"])



# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence, RunnableParallel

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Generate a tweet about {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Generate a Linkedin post about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# parallel_chain = RunnableParallel({
#     'tweet': RunnableSequence(prompt1, model, parser),
#     'linkedin': RunnableSequence(prompt2, model, parser)
# })

# result = parallel_chain.invoke({'topic':'AI'})

# print(result['tweet'])
# print(result['linkedin'])

