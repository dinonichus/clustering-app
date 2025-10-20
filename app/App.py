import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import Home
import Dashboard
import About



st.set_page_config(page_title="Analisis Harga Pangan", layout="wide")


col_kiri, col_menu, col_kanan = st.columns([1, 3, 1]) 
with col_menu:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Dashboard", "About"],
        icons=["house", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0",
                "background-color": "#eef1f5ff", 
                "justify-content": "center",
                "align-text": "center",
                "border-radius": "100px",
            },
            "icon": {
                # "color": "#a9abb1ff", 
                "font-size": "18px"},
            "nav-link": {
                "border-radius": "100px",
                "font-size": "16px", 
                "text-align": "center", 
                "color": "#a9abb1ff", 
                "background-color": "#eef1f5ff",
                },
            "nav-link-selected": {
                "border-radius": "100px",
                "color": "#000516ff",
                "font-weight": "normal",
                "icon": {"color": "#000516ff"},
                },
        }
    )

if selected == "Home":
    Home.app()
elif selected == "Dashboard":
    Dashboard.app()
elif selected == "About":
    About.app()
