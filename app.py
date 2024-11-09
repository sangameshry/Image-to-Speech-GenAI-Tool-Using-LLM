import os
import time
from typing import Any
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from transformers import pipeline
import json
from utils.custom import css_code

# Define your NIM API key directly in the code
NIM_API_KEY = st.secrets['NIM_API_KEY']
HUGGINGFACE_API_TOKEN = st.secrets['HUGGINGFACE_API_TOKEN']


def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


def generate_text_from_image(url: str) -> str:
    image_to_text: Any = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    generated_text: str = image_to_text(url)[0]["generated_text"]

    print(f"IMAGE INPUT: {url}")
    print(f"GENERATED TEXT OUTPUT: {generated_text}")
    return generated_text


def generate_story_from_text(scenario: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"You are a talented storyteller who can create a story from a simple narrative. Create a story using the following scenario; the story should be a maximum of 50 words long: {scenario}"
        }
    ]

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    payload = {
        "model": "meta/llama3-70b-instruct",
        "max_tokens": 1024,
        "stream": False,
        "temperature": 0.5,
        "top_p": 1,
        "stop": None,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "seed": 0,
        "messages": messages
    }
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    print("Raw response:", response.text)  # Inspect the raw response

    if response.ok:
        try:
            response_json = response.json()
            generated_story = response_json['choices'][0]['message']['content']  # Correct extraction
        except json.JSONDecodeError as e:
            generated_story = f"JSON Decode Error: {str(e)}"
    else:
        generated_story = f"Error: Received status code {response.status_code}, {response.text}"

    print(f"TEXT INPUT: {scenario}")
    print(f"GENERATED STORY OUTPUT: {generated_story}")
    return generated_story

def generate_speech_from_text(message: str) -> Any:
    API_URL: str = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers: dict[str, str] = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads: dict[str, str] = {
        "inputs": message
    }

    response: Any = requests.post(API_URL, headers=headers, json=payloads)
    with open("generated_audio.flac", "wb") as file:
        file.write(response.content)


def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼")
    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.image("audio-img/logo.jpg")
        st.write("---")
        st.write("AI App created by @ Sangamesh R Y and Tanisha")

    st.header("Image-to-Story Converter")
    uploaded_file: Any = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data: Any = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Progress bar animation
        progress_bar(100)
        
        # Generate text description from image
        scenario: str = generate_text_from_image(uploaded_file.name)
        
        # Generate story from text
        story: str = generate_story_from_text(scenario)
        
        # Convert story to audio
        generate_speech_from_text(story)

        with st.expander("Generated Image scenario"):
            st.write(scenario)
        with st.expander("Generated short story"):
            st.write(story)

        st.audio("generated_audio.flac")


if __name__ == "__main__":
    main()
