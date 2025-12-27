from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os


load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=os.getenv("GEMINI_API_KEY"))

chat_history = [
    SystemMessage(content='You are a helpful AI assistant for humans.')
]

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        print("\nExiting chat...")
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

# Show the full chat history
print("\nFull Chat History:")
for message in chat_history:
    role = type(message).__name__.replace("Message", "")  # System, Human, AI
    print(f"{role}: {message.content}")



# using langchain

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

# chat_history = [
#     SystemMessage(content='You are a helpful AI assistant')
# ]

# while True:
#     user_input = input('You: ')
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input == 'exit':
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content))
#     print("AI: ",result.content)

# print(chat_history)