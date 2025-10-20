import streamlit as st
from Utils import display_lottie_home

def app():
    st.write("#")
    
    st.markdown("""
        <h1 style='text-align: center; font-size:65px; color: #2f5088ff; line-height: 0.5; margin-bottom: 0; padding-top: 0px'>Pengelompokan dan Analisis</h1>
        <h1 style='text-align: center; font-size:65px; color: #000000; line-height: 0.5; margin-top: 0; margin-bottom: 0;'>Harga Pangan di Pasar Tradisional</h1>
        <h3 style='text-align: center; font-size:45px; color: #000000; line-height: 1; margin-top: 0; margin-botton: 0'>Wilayah Indonesia Bagian Timur</h3>
        """, unsafe_allow_html=True)
        
    st.write("""
        <div style="text-align: center; font-size:18px; padding-top: 10px">
            Penenerapan algoritma <b>K-Means</b> dan <b>DBSCAN</b> 
            untuk menganalisis pola harga pangan.<br><br>
        </div>
        """, unsafe_allow_html=True)
    
    # DASHBOARD_PAGE = "pages/Dashboard.py"
    # ABOUT_PAGE = "pages/About.py"

    # spacer1, col_dashb, col_about, spacer2 = st.columns([1, 0.2, 0.2, 1])
    # with col_dashb:
    #     if st.button("Clustering"):
    #         st.switch_page(DASHBOARD_PAGE)
    # with col_about:        
    #     if st.button("About"):
    #         st.switch_page(ABOUT_PAGE)

    lottie_placeholder = st.empty()
    display_lottie_home(lottie_placeholder, height=220, width=500, key="home")