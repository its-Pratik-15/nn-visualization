import time 
import random 

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.map([{"lat":40,"lon":70}])

progress_bar = st.progress(0)

# st.

for i in range(100):
    progress_bar.progress(i + 1)


number = st.number_input("Insert a number")

text_field = st.text_input("Insert some text")