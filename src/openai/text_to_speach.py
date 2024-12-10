
import openai
import streamlit as st
import base64
from src.keys.keys import get_api_key

# Initialize OpenAI client
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def generate_audio(text, voice):
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4o-audio-preview",
        #     messages=[{"role": "user", "content": text}],
        #     audio={"voice": voice, "format": "wav"},
        # )
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
            )
        # audio_data = response.choices[0].message["audio"]["data"]
        # audio_data = response
        # wav_bytes = base64.b64decode(audio_data)
        # Save the audio file temporarily
        # output_path = "/content/output.wav"
        # with open(output_path, "wb") as f:
        #     f.write(wav_bytes)
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    
def create_voice_dropdown():

# List of options
    options = ["alloy", "clarity", "warmth", "energy"]

    # Create a dropdown
    selected_option = st.selectbox("Choose an option:", options)

    # Display the selected option
    st.write("You selected:", selected_option)
    return selected_option