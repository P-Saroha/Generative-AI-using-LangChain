## conditional chain using PydanticOutputParser, RunnableBranch, RunnableLambda
# This code demonstrates how to create a conditional chain using LangChain with Google Gemini model.
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# Load environment variables
load_dotenv()

# Use Gemini via API key
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Output parser for string
parser = StrOutputParser()

# Pydantic model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Output parser for structured sentiment classification
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt to classify sentiment
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n{feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Sentiment classification chain
classifier_chain = prompt1 | model | parser2

# Prompts for positive and negative responses
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n{feedback}',
    input_variables=['feedback']
)

# Conditional branching
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# Final chain
chain = classifier_chain | branch_chain

# Run the chain
print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# Visualize the chain graphically
chain.get_graph().print_ascii()


## Coniditional Chain Example with help of PydanticOutputParser, RunnableBranch, RunnableLambda 
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal

# load_dotenv()

# model = ChatOpenAI()

# parser = StrOutputParser()

# class Feedback(BaseModel):

#     sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# parser2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
#     input_variables=['feedback'],
#     partial_variables={'format_instruction':parser2.get_format_instructions()}
# )

# classifier_chain = prompt1 | model | parser2

# prompt2 = PromptTemplate(
#     template='Write an appropriate response to this positive feedback \n {feedback}',
#     input_variables=['feedback']
# )

# prompt3 = PromptTemplate(
#     template='Write an appropriate response to this negative feedback \n {feedback}',
#     input_variables=['feedback']
# )

# branch_chain = RunnableBranch(
#     (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
#     (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )

# chain = classifier_chain | branch_chain

# print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# chain.get_graph().print_ascii()