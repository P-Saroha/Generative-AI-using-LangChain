import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini 1.5 Flash model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Start chat without system role
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["You are a helpful assistant."]
    }
])

# User message to the chat
prompt = "Please summarize the paper titled 'A Comprehensive Study on Generative Models' with a formal tone and a length of 200 words."

# Send the message
response = chat.send_message(
    prompt,
    generation_config={
        "max_output_tokens": 200,
        "temperature": 0.3
    }
)

# Print the model's response
print(response.text)



## with langchain 
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

# messages=[
#     SystemMessage(content='You are a helpful assistant'),
#     HumanMessage(content='Tell me about LangChain')
# ]

# result = model.invoke(messages)

# messages.append(AIMessage(content=result.content))

# print(messages)
