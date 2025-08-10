from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
import os

# Load API key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Prompt to write a joke
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Output parser
parser = StrOutputParser()

# Prompt to explain a joke
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

# Step 1: Generate joke
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# Step 2: Pass joke to both original output and explanation
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})

# Step 3: Combine both
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Execute
print(final_chain.invoke({"topic": "cricket"}))


# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# prompt2 = PromptTemplate(
#     template='Explain the following joke - {text}',
#     input_variables=['text']
# )

# joke_gen_chain = RunnableSequence(prompt1, model, parser)

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'explanation': RunnableSequence(prompt2, model, parser)
# })

# final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# print(final_chain.invoke({'topic':'cricket'}))