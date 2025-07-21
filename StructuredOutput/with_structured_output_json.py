import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

#  Setup Gemini model (1.5 Flash)
# model = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

## if gemini_api_key will not work from .env file, you can replace it with your key directly
# and if work the uncommented code above and comment the below line

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key="Replace with your key"  # Replace with your key
)

#  JSON schema for structured output
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative or positive"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

#  Prompt for structured output
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured data from the review based on the provided schema."),
    ("human", "{input}")
])

#  Chain model with prompt and structured output parser
chain = prompt | model.with_structured_output(schema=json_schema)

#  Input review
review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by admin 11122
"""

#  Invoke model
result = chain.invoke({"input": review_text})

#  Output result
print(result)










## using openai with structured output
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from typing import TypedDict, Annotated, Optional, Literal
# from pydantic import BaseModel, Field

# load_dotenv()

# model = ChatOpenAI()

# # schema
# json_schema = {
#   "title": "Review",
#   "type": "object",
#   "properties": {
#     "key_themes": {
#       "type": "array",
#       "items": {
#         "type": "string"
#       },
#       "description": "Write down all the key themes discussed in the review in a list"
#     },
#     "summary": {
#       "type": "string",
#       "description": "A brief summary of the review"
#     },
#     "sentiment": {
#       "type": "string",
#       "enum": ["pos", "neg"],
#       "description": "Return sentiment of the review either negative, positive or neutral"
#     },
#     "pros": {
#       "type": ["array", "null"],
#       "items": {
#         "type": "string"
#       },
#       "description": "Write down all the pros inside a list"
#     },
#     "cons": {
#       "type": ["array", "null"],
#       "items": {
#         "type": "string"
#       },
#       "description": "Write down all the cons inside a list"
#     },
#     "name": {
#       "type": ["string", "null"],
#       "description": "Write the name of the reviewer"
#     }
#   },
#   "required": ["key_themes", "summary", "sentiment"]
# }


# structured_model = model.with_structured_output(json_schema)

# result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

# The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

# However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

# Pros:
# Insanely powerful processor (great for gaming and productivity)
# Stunning 200MP camera with incredible zoom capabilities
# Long battery life with fast charging
# S-Pen support is unique and useful
                                 
# Review by Nitish Singh
# """)

# print(result)