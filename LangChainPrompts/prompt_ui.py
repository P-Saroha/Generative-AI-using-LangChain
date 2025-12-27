import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load your .env file
load_dotenv()

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit UI
st.title(" Research Paper Summarizer (Gemini Flash)")

paper_input = st.selectbox(
    " Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    " Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    " Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load the prompt template from template.json
def load_prompt_template(path):
    with open(path, 'r') as f:
        return json.load(f)["template"]

# Fill the prompt with user inputs
def build_prompt(template, paper, style, length):
    return template.format(
        paper_input=paper,
        style_input=style,
        length_input=length
    )

# Load the prompt from file
template_str = load_prompt_template('template.json')

# Handle button click
if st.button(" Summarize"):
    final_prompt = build_prompt(template_str, paper_input, style_input, length_input)

    try:
        #  Use fast Gemini model
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

        # Generate content
        response = model.generate_content(final_prompt)

        # Display response
        st.subheader(" Generated Summary")
        st.write(response.text)

    except Exception as e:
        st.error(f"Error: {str(e)}")


## Using LangChain with OpenAI for Research Paper Summarization
## This code snippet demonstrates how to use LangChain with OpenAI to summarize research papers based on user

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_core.prompts import PromptTemplate,load_prompt
# load_dotenv()
# model = ChatOpenAI()

# st.header('Reasearch Tool')

# paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

# style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

# length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template = load_prompt('template.json')



# if st.button('Summarize'):
#     chain = template | model
#     result = chain.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     })
#     st.write(result.content)
