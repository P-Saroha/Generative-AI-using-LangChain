from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=128)

documents = [
    "delhi is capital of india",
    "paris is capital of france",
    "new york is capital of US"
]

result = embedding.embed_query(documents)

print(str(result))


