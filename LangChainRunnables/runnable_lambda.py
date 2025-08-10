from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
import os

# Load API key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Custom function to count words
def word_count(text):
    return len(text.split())

# Prompt for joke
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Output parser
parser = StrOutputParser()

# Step 1: Generate joke
joke_gen_chain = RunnableSequence(prompt, model, parser)

# Step 2: Pass joke & count words in parallel
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

# Step 3: Combine
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Run
result = final_chain.invoke({"topic": "AI"})

# Format output
final_result = f"""{result['joke']} 
word count - {result['word_count']}"""

print(final_result)



# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

# load_dotenv()

# def word_count(text):
#     return len(text.split())

# prompt = PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# joke_gen_chain = RunnableSequence(prompt, model, parser)

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(word_count)
# })

# final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# result = final_chain.invoke({'topic':'AI'})

# final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# print(final_result)