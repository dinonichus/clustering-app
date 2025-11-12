import streamlit as st
import math
import folium
import pandas as pd
import seaborn as sns
from streamlit_folium import st_folium

def show_map():

    df_map = pd.read_csv('../misc/mapping.csv')

    if {"Latitude", "Longitude"}.issubset(df_map.columns):
        # Koordinat rata-rata untuk menentukan posisi awal peta
        avg_lat = df_map["Latitude"].mean()
        avg_lon = df_map["Longitude"].mean()

        m = folium.Map(
            location=[avg_lat, avg_lon+5], 
            zoom_start=5.2,
        )

        for _, row in df_map.iterrows():
            color = "red"

            popup_html = f"<b>{row['Kabupaten/Kota']}</b>"
        
            popup_obj = folium.Popup(
                html=popup_html,
                max_width=500, # Mengatur lebar maksimum
                min_width=100  # Mengatur lebar minimum (untuk lebar tetap)
                # Catatan: Tinggi seringkali dikontrol oleh konten, tapi bisa ditambahkan di style jika perlu.
            )

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=popup_obj
            ).add_to(m)

        st_folium(
            m, 
            width=750, 
            height=500, 
            key='folium_cluster_map_stable', 
            returned_objects=[] 
        )

    else:
        print("test")
        st.warning("Kolom Latitude dan Longitude belum tersedia pada data hasil penggabungan.")

def app():
    kabupaten_kota_list = [
        "Kab. Banggai",
        "Kab. Bulukumba",
        "Kab. Gorontalo",
        "Kab. Lombok Timur",
        "Kab. Majene",
        "Kab. Manokwari",
        "Kab. Merauke",
        "Kab. Mimika",
        "Kab. Polewali Mandar",
        "Kab. Sumba Timur",
        "Kab. Sumbawa",
        "Kota Ambon",
        "Kota Bau-Bau",
        "Kota Bima",
        "Kota Denpasar",
        "Kota Gorontalo",
        "Kota Jayapura",
        "Kota Kendari",
        "Kota Kotamobagu",
        "Kota Kupang",
        "Kota Makassar",
        "Kota Mamuju",
        "Kota Manado",
        "Kota Mataram",
        "Kota Maumere",
        "Kota Palopo",
        "Kota Palu",
        "Kota Parepare",
        "Kota Singaraja",
        "Kota Sorong",
        "Kota Ternate",
        "Kota Tual",
        "Kota Watampone"
    ]

    size = math.ceil(len(kabupaten_kota_list) / 3) 

    col1_data = kabupaten_kota_list[0:size]
    col2_data = kabupaten_kota_list[size:size*2]
    col3_data = kabupaten_kota_list[size*2:]

    komoditas_pangan = [
        "Ayam",
        "Bawang Merah",
        "Bawang Putih",
        "Beras",
        "Cabai Merah",
        "Cabai Rawit",
        "Gula",
        "Minyak",
        "Sapi",
        "Telur"
    ]

    size2 = math.ceil(len(komoditas_pangan) / 3)
    col1_data1 = komoditas_pangan[0:size2]
    col2_data1 = komoditas_pangan[size2:size2*2]
    col3_data1 = komoditas_pangan[size2*2:]

    st.title("""ℹ️ About""")
    st.write("---")

    st.write("#### Pengelompokan dan Analisis Harga Pangan di Pasar Tradisional Wilayah Indonesia bagian Timur dengan K-Means dan DBSCAN")
    st.write("Aplikasi ini ditujukan sebagai program perancangan tugas akhir yang dapat mengelompokan wilayah Indonesia bagian Timur berdasarkan harga komoditas pangan melalui penerapan K-Means dan DBSCAN.")

    st.write("")
    st.write("#### Kawasan Timur Indonesia")
    col_map, col_left, col_middle, col_right = st.columns([1, 0.3, 0.3, 0.3])
    with col_map:
        show_map()
    with col_left:
        for anggota in col1_data:
            st.write(f"- {anggota}")

    with col_middle:
        for anggota in col2_data:
            st.write(f"- {anggota}")

    with col_right:
        for anggota in col3_data:
            st.write(f"- {anggota}")

    st.write("")
    st.write("#### Komoditas Pangan")
    st.write("Komoditas pangan yang digunakan pada perancangan ini meliputi:")
    col_a, col_b, col_c, space = st.columns([0.2, 0.2, 0.2, 0.5])
    with col_a:
        for anggota in col1_data1:
            st.write(f"- {anggota}")

    with col_b:
        for anggota in col2_data1:
            st.write(f"- {anggota}")

    with col_c:
        # st.write(col3_data1)
        for anggota in col3_data1:
            st.write(f"- {anggota}")


    st.write("---")
    st.write("#### Sumber Data")
    st.write("Dataset harga komoditas pangan diperoleh dari situs resmi [Pusat Inforasi Harga Pangan Strategis Nasional](https://www.bi.go.id/hargapangan/) yang dikelola oleh Bank Indonesia.")

    st.write("")
    st.write("#### Metode Clustering")
    st.write("""
    Terdapat dua metode clustering yang digunakan, yaitu:
    1. K-Means: Pengemlompokan data dilakukan berdasarkan jarak terdekat terhadap titik pusatnya melalui penentuan jumlah cluster. Data yang saling berdekatan dengan suatu titik pusat akan digolongkan menjadi satu kelopok.
    2. DBSCAN (Density Based Spacial Clustering of Application with Noise): Pengelompokan data berdasarkan kepadatan atau kerapatan jarak antar data. Data-data yang saling berdekatan dan saling bertetangga akan digolongkan menjadi satu kelompok melalui penentuan nilai batas jarak maksimum (epsilon) dan jumlah minimum tetangga (minPts).""")

    st.write("---")
    st.write("#### 📢 Contact Us")
    st.write("**Gabriella Ignatia**")

    col_1, col_2 = st.columns(2)

    with col_1:
        colA_, col_, colB_ = st.columns([0.08, 0.001, 0.5])
        with colA_:
            st.write("E-mail:")
            st.write("Github:")
            st.write("LinkedIn:")

        with colB_:
            st.write("gaby.ignatia@gmail.con")
            st.write("[dinonichus](https://github.com/dinonichus)")
            st.write("[Gabriella Ignatia](https://www.linkedin.com/in/gabriella-ignatia-1ba897333/)")

    st.write("---")
    st.write("Teknik Informatika, [Fakultas Teknologi Informasi](https://fti.untar.ac.id/)")
    st.write("Universitas Traumanagara")
    st.write("""Jln Letjen S. Parman No. 1, Grogol Petamburan
                Jakarta Barat, 11440
                Gedung R, Lt. 11, Kampus I""")