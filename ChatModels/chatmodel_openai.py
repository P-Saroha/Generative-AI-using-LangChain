from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-2024-08-06',
                   temperature = 1.0,
                   max_tokens = 100,
                   top_p = 1.0,
                   frequency_penalty = 0.0,
                   presence_penalty = 0.0)
result = model.invoke('what is the capital of india?')
print(result.content)  # Access the content of the response