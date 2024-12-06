# Import necessary libraries
import streamlit as st
import os
from together import Together
from src.keys.keys import get_api_key
from src.together.llama.generate_code import generate_code_with_codellama

# get the api key
together_api_key = get_api_key('TOGETHER_API_KEY')    

# Initialize Together client
client = Together()

# st.set_page_config(
#     page_title="Jhirley Fonte, Projects",
#     page_icon="👋"
# )

# Streamlit app layout
st.title("Python Code Generator with CodeLlama")
st.write("Enter a description of the Python application or code you need. CodeLlama will generate the corresponding Python code.")



# Input box for the user to enter a description
description = st.text_area("Application or Code Description", placeholder="Describe the application or code you want")

# Button to trigger code generation
if st.button("Generate Code"):
    if description.strip():
        st.write("### Generated Python Code")
        # Generate code
        generated_code = generate_code_with_codellama(client, description)
        st.code(generated_code, language="python")
    else:
        st.error("Please provide a valid description.")


st.write("\nhttps://drlee.io/building-a-python-code-generator-with-codellama-in-streamlit-cloud-4a78886918eb")