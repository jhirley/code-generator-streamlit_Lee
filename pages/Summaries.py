import streamlit as st
import re
## setting up the language model
from langchain_together import ChatTogether
## import the youtube documnent loader from LangChain
from langchain_community.document_loaders import YoutubeLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from src.keys.keys import get_api_key


# Access the environment variable
together_api_key = get_api_key('TOGETHER_API_KEY')

# st.set_page_config(
#     page_title="Jhirley Fonte, Projects",
#     page_icon="👋",
# )

# Streamlit app layout
st.title("Video Summaries")
st.write("This will take the youtube URL of a video and provide a summary of the transcript.")

# sidebar()

# Create a slider with values between 0 and 10
creativity_slider = st.sidebar.slider(
    "Creativity",
    min_value=0,
    max_value=10,
    value=1,
    step=1,
    key="creativity"
)

# Display the current value of the slider
st.write(f"Current creativity level: {creativity_slider}")
if creativity_slider == 0:
    hallucinations_score = 0
else:
    hallucinations_score = creativity_slider * 0.1

# Create a text input field for the user to enter the YouTube video URL
video_url = st.text_input("Enter the YouTube video URL:")
# url_pattern = r'^(https?:\/\/)?([\w\-]+\.)+[\w\-]+(\/[\w\-.,@?^=%&:\/~+#]*)?$'
url_pattern = r'^(https?:\/\/)?([\w\-]+\.)+[\w\-]+(\/[\w\-.,@?^=%&:\/~+#]*)?$'
if re.match(url_pattern, video_url) is None:
    st.stop()

# video_url = 'https://www.youtube.com/watch?v=q4Dmq2TKNq4'

if st.button("Generate Summary"):
    
    llm = ChatTogether(api_key=together_api_key, temperature=hallucinations_score, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    data = loader.load()

    description_template = PromptTemplate(
        input_variables=["video_transcript"],
        template="""
        Read through the entire transcript carefully.
            Provide a concise summary of the video's main topic and purpose.
            Extract and list the five most interesting or important points from the transcript. 
            For each point: State the key idea in a clear and concise manner.

            - Ensure your summary and key points capture the essence of the video without including unnecessary details.
            - Use clear, engaging language that is accessible to a general audience.
            - If the transcript includes any statistical data, expert opinions, or unique insights, 
            prioritize including these in your summary or key points.
        
        Video transcript: {video_transcript}    """
    )

    # Create the LLMChain with the prompt template and the LLM
    chain = LLMChain(prompt=description_template, llm=llm)

    # Run the chain with the provided video transcript
    summary = chain.invoke({
        "video_transcript": data[0].page_content
    })

    # Access the content attribute of the AIMessage object
    st.write(summary['text'])
    st.divider()
    st.write(summary["video_transcript"])


st.write("\nhttps://drlee.io/building-a-real-application-with-langchain-and-together-ai-6a09fcf54c97")