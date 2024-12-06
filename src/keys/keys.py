import os
import streamlit as st
from dotenv import load_dotenv
from functools import lru_cache

# @lru_cache()
def get_api_key(key_name):
    load_dotenv()
    if isinstance(key_name, list):
        keys = {}
        for key in key_name:
            api_key = os.getenv(key)
            if api_key:
                keys[key] = api_key
            else:
                keys[key] = st.secrets[key]
        return keys
    elif isinstance(key_name, str):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
        else:
            return st.secrets[key_name]

# Load environment variables from .env file

# Access the environment variable
# together_api_key = os.getenv('TOGETHER_API_KEY')

# if together_api_key:
#     # st.secrets["TOGETHER_API_KEY"] = together_api_key
#     os.environ['TOGETHER_API_KEY'] = together_api_key
# else:
#     os.environ['TOGETHER_API_KEY'] = st.secrets["TOGETHER_API_KEY"]
#     together_api_key = st.secrets["TOGETHER_API_KEY"]