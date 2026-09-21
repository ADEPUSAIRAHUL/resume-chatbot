import os
import streamlit as st
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

st.title("Resume Chatbot")

question = st.text_input("Ask a question about my resume:")

if question:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful resume assistant. "
                    "Answer clearly and in detail. "
                    "Do not invent information."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=4096
    )

    st.write(response.choices[0].message.content)