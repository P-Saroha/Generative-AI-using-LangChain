from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv('Generative-AI-using-LangChain\\.env')

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=128)

result = embedding.embed_query("capital of UK?")

print(str(result))


