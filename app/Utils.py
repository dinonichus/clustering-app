import json
import streamlit as st
from streamlit_lottie import st_lottie

@st.cache_data
def lottie_file():
    with open('../misc/Livebot.json', "r") as f:
        return json.load(f)
    
@st.cache_data
def lottie_file2():
    with open('../misc/blink.json', "r") as f:
        return json.load(f)  
    
def display_loading_lottie(placeholder, height=100, width=100, key="loading_key"):
    lottie_data = lottie_file()
    if lottie_data:
        with placeholder:
            col_spacer_left, col_lottie, col_spacer_right = st.columns([1, 1, 1])

            with col_lottie:

                 st_lottie(
                    lottie_data,
                    speed=1.5,
                    loop=True,
                    quality='high',
                    height=height,
                    width=width,
                    key=key 
                 )

def display_lottie_bot(placeholder, height=100, width=500, key="loading_key"):
    lottie_data = lottie_file2()
    if lottie_data:
        with placeholder:

                 st_lottie(
                    lottie_data,
                    speed=1.5,
                    loop=True,
                    quality='high',
                    height=height,
                    width=width,
                    key=key 
                 )

def display_lottie_home(placeholder, height=100, width=100, key="loading_key"):
    lottie_data = lottie_file2()
    if lottie_data:
        with placeholder:
            col_spacer_left, col_lottie, col_spacer_right = st.columns([1, 1, 1])

            with col_lottie:

                 st_lottie(
                    lottie_data,
                    speed=1.5,
                    loop=True,
                    quality='high',
                    height=height,
                    width=width,
                    key=key 
                 )
