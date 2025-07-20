import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import genai
from dotenv import load_dotenv
import os 

load_dotenv()

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")


