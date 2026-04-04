# streamlit_app.py

import streamlit as st
from app import summarize_dialogue# import the function from your FastAPI script

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("📝 Text Summarizer App")

st.write("Enter your text below and click **Summarize** to get a concise summary.")

text_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize!")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_dialogue(text_input)
        st.subheader("Summary:")
        st.write(summary)

import streamlit as st
from streamlit.components.v1 import html

html(open("templates/index.html").read(), height=600)