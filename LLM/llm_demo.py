from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo-instruct",
)

result = llm.invoke('what is the capital of india?')
print(result)
        