## Integrate code with OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

import streamlit as st

# streamlit framework

st.title("Celebrity Search Results")
input_text = st.text_input("Search the Celebrity You want!!")

## Prompt Templates - Custom Search

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)
 # Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
event_memory = ConversationBufferMemory(input_key='dob', memory_key='event_history')


## Open LLMS
llm = OpenAI(temperature = 0.8)
## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain1=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='person', memory= person_memory)


# Prompt Templates

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob', memory= dob_memory)

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Events happends around {dob}"
)

chain3=LLMChain(
    llm=llm,prompt=third_input_prompt,verbose=True,output_key='events', memory= event_memory)


parent_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['name'],output_variables=['person','dob','events'],verbose=True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'): 
         st.info(person_memory.buffer)
    with st.expander('Major Events'): 
        st.info(event_memory.buffer)