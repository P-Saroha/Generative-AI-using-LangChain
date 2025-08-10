from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
import os

# Load API key from .env
load_dotenv()

# Map GEMINI_API_KEY to the variable LangChain expects
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# First prompt
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Output parser
parser = StrOutputParser()

# Second prompt
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

# Chain sequence
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# Run the chain
print(chain.invoke({"topic": "AI"}))


# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# prompt2 = PromptTemplate(
#     template='Explain the following joke - {text}',
#     input_variables=['text']
# )

# chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# print(chain.invoke({'topic':'AI'}))