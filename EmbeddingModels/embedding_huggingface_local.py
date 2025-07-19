from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "delhi is capital of india"

vector =  embedding.embed_query(text)

print(str(vector))