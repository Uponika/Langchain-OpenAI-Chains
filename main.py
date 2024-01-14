## Integrate code with OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = openai_key

import streamlit as st

# streamlit framework

st.title("Langchain Demo with OpenAI")
input_text = st.text_input("Search the Topic You want!!")

## Open LLMS
llm = OpenAI(temperature = 0.8)

if input_text:
    st.write(llm(input_text))