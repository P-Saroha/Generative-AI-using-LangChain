from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
import os

# Load API key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Prompt to generate detailed report
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt to summarize
prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)

# Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Output parser
parser = StrOutputParser()

# Step 1: Generate report
report_gen_chain = prompt1 | model | parser

# Step 2: Branch — summarize if report is longer than 300 words, else return as is
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

# Step 3: Combine
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Run
print(final_chain.invoke({"topic": "Russia vs Ukraine"}))




# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Write a detailed report on {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Summarize the following text \n {text}',
#     input_variables=['text']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# report_gen_chain = prompt1 | model | parser

# branch_chain = RunnableBranch(
#     (lambda x: len(x.split())>300, prompt2 | model | parser),
#     RunnablePassthrough()
# )

# final_chain = RunnableSequence(report_gen_chain, branch_chain)

# print(final_chain.invoke({'topic':'Russia vs Ukraine'}))


