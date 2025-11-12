from datetime import datetime
from functools import reduce
import hashlib
import math
import os
import re
import tempfile
import folium
from fpdf import FPDF
from PIL import Image
from streamlit_folium import st_folium
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import traceback
import time
from kneed import KneeLocator
import matplotlib.dates as mdates 
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from streamlit_lottie import st_lottie
from Utils import display_loading_lottie, display_lottie_bot

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples


TEMPLATE_FILE_PATH = "../dataset/template_data.xlsx"
READY_FILE_PATH = '../dataset/mentah/semua_komoditas.xlsx'

@st.cache_data
def dbscan_test(df):
    # X = df.drop("Kabupaten/Kota", axis=1)
    X = data_scaling(df)
    # st.write(X)

    k = 2
    neigh = NearestNeighbors(n_neighbors=k) 
    distances = neigh.fit(X).kneighbors()[0]

    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances, axis=0)

    kneedle = KneeLocator(
        range(len(k_distances)), 
        k_distances, 
        S=1.0, 
        curve='convex', 
        direction='increasing'
    )

    eps_recommend = kneedle.knee_y
    eps_recommend = round(eps_recommend, 2)
    # st.write(f"### Rekomendasi Epsilon Otomatis: **{eps_recommend}**")

    if eps_recommend - 3 < 0:
        start_eps = 0.05
    else:
        start_eps = eps_recommend - 3
    end_eps = eps_recommend + 3

    param_grid = {
        "eps": np.arange(start_eps, end_eps, 0.05),
        "min_pts": list(range(2, 11))
    }
    results = []

    for eps in param_grid["eps"]:
        # st.write(f"#### Mencoba Epsilon: **{eps}**")
        for min_pts in param_grid["min_pts"]:
            start_time = time.time()
            model = DBSCAN(eps=eps, min_samples=min_pts)
            labels = model.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            if n_clusters > 1:
                mask = labels != -1
                X_masked = X[mask]
                labels_masked = labels[mask]

                sil_score, dbi = evaluation(X_masked, labels_masked)

            else:
                sil_score = -1
                dbi = np.inf

            runtime = time.time() - start_time
            results.append({
                "eps": eps,
                "min_pts": min_pts,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": sil_score,
                "dbi": dbi,
                "runtime_sec": runtime
            })

    # HASIL EKSPERIMEN
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["silhouette", "n_noise", "dbi"], ascending=[False, True, True])
    print("### Hasil Eksperimen DBSCAN (Top 10) ###")
    print(results_df.head(1))

    # PARAMETER TERBAIK
    best_params = results_df.iloc[0]
    best_eps = best_params["eps"]
    best_min_pts = int(best_params["min_pts"])
    print(f"\nParameter Terbaik: eps={best_eps}, min_pts={best_min_pts}")
    # st.write(f"### Rekomendasi Epsilon Otomatis: **{best_eps}**")

    return best_params


@st.cache_data
def kmeans_test(df):
    # X = df.drop("Kabupaten/Kota", axis=1)
    # st.write(X)
    X = data_scaling(df)

    param_grid = {
        "n_clusters": list(range(2, 11)),  
    }
    n_init = 10
    results = []

    for n_clusters in param_grid["n_clusters"]:
        if n_clusters > 1:
            start_time = time.time()
            model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
            labels = model.fit_predict(X)

            if len(set(labels)) > 1:
                sil_score, dbi = evaluation(X, labels)
            else:
                sil_score = -1
                dbi = np.inf
            
            runtime = time.time() - start_time
            results.append({
                "n_clusters": n_clusters,
                "n_init": n_init,
                "silhouette": sil_score,
                "dbi": dbi,
                "runtime_sec": runtime
            })

    results_df = pd.DataFrame(results)
    best_params = results_df.iloc[0]

    # print("BEST PARAM KMEANS-----------------------", best_params)
    return best_params
    

@st.cache_data
def kmeans_clustering(df, n_clusters):
    # X = df.drop("Kabupaten/Kota", axis=1)
    X = data_scaling(df)
    
    start = time.time()
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    df_cluster = df[['Kabupaten/Kota']].copy()
    df_cluster['Cluster'] = labels

    sil, dbi = evaluation(X, labels)
    runtime = time.time() - start
    
    return df_cluster, sil, dbi, runtime


@st.cache_data
def dbscan_clustering(df, eps, min_pts):
    X = data_scaling(df)
    print("-----------\n", eps, min_pts)

    start_time = time.time()
    model = DBSCAN(eps=eps, min_samples=min_pts)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    if n_clusters > 1:
        mask = labels != -1
        X_masked = X[mask]
        labels_masked = labels[mask]

        sil, dbi = evaluation(X_masked, labels_masked)

    else:
        sil = -1
        dbi = np.inf

    runtime = time.time() - start_time
    df_cluster = df[['Kabupaten/Kota']].copy()
    df_cluster['Cluster'] = labels

    print("DBSCAN: ", sil, dbi)
    return df_cluster, sil, dbi, runtime


@st.cache_data
def evaluation(X, labels):
    mask = labels != -1
    X_masked = X[mask]
    labels_masked = labels[mask]

    if len(set(labels_masked)) <= 1:
        return -1, np.inf
    
    sil_score = silhouette_score(X_masked, labels_masked)
    dbi = davies_bouldin_score(X_masked, labels_masked)
    return sil_score, dbi


def display_test(df_kmeans, df_dbscan):
    kmeans_winner = ""
    dbscan_winner = ""
    if(df_kmeans['silhouette'] > df_dbscan['silhouette']):
        kmeans_winner = "🏆 "
    else:
        dbscan_winner = "🏆 "

    col_kmeans, col_dbscan = st.columns([1, 1])

    with col_kmeans:
        st.subheader(kmeans_winner + "K-MEANS CLUSTERING")
        st.metric("Silhouette Score", f"{df_kmeans['silhouette']:.4f}")
        st.metric("Davies-Bouldin Index", f"{df_kmeans['dbi']:.4f}")
        st.metric("Runtime", f"{df_kmeans['runtime_sec']:.4f}s")

        st.write(f"### **📈 Parameter Terbaik**")
        col1, col2, _ = st.columns([0.2, 0.2, 0.6])
        with col1:
            st.write(f"##### **n_clusters**")
        with col2:
            st.write(f"##### **: {int(df_kmeans['n_clusters'])}**")

    with col_dbscan:
        st.subheader(dbscan_winner + "DBSCAN CLUSTERING")

        if df_dbscan['silhouette'] == -1 or df_dbscan['dbi'] == np.inf:
            st.write(f"##### **DBSCAN gagal untuk membentuk cluster!**")

        else:
            col1, col2, _ = st.columns([0.5,0.5, 0.5])
            with col1:
                st.metric("Silhouette Score", f"{df_dbscan['silhouette']:.4f}")
                st.metric("Davies-Bouldin Index", f"{df_dbscan['dbi']:.4f}")
                st.metric("Runtime", f"{df_dbscan['runtime_sec']:.4f}s")
            with col2:
                st.metric("Number of clusters", f"{int(df_dbscan['n_clusters'])}")
                st.metric("Number of noise points", f"{int(df_dbscan['n_noise'])}")

            st.write(f"### **📈 Parameter Terbaik**")
            col1, col2, _ = st.columns([0.2, 0.2, 0.6])
            with col1:
                st.write(f"##### **Epsilon**")
                st.write(f"##### **Minimum Points**")
            with col2:
                st.write(f"##### **: {df_dbscan['eps']:.2f}**")
                st.write(f"##### **: {int(df_dbscan['min_pts'])}**")

    st.info("Silhouette Score mendekati 1 menandakan anggota cluster sudah berada di cluster yang tepat.")
    st.info("Davies-Bouldin Index mendekati 0 menandakan kerapatan jarak tiap anggota dalam cluster dan pemisahan yang jelas antar cluster.")



def display_data(df):
    st.dataframe(df)
    st.write(f"**Dimensi Data:** {df.shape[0]} rows and {df.shape[1]} columns.")

    null_count = df.isna().sum() 
    null_series = null_count[null_count > 0]

    if null_series.empty:
        st.write(f"**Total null:** 0.")
    else:
        st.write(f"**Total null:** {null_series.sum()}.")


@st.cache_data
def data_cleaning(df_input):
    df = df_input.copy()
    # kekosongan sebanyak lebih dari 40%, kabupaten akan dihapus
    PERCENT_THRESHOLD = 0.4  

    column_name = df.columns[0]
    new_column_name = str(column_name).strip() 
    df.rename(columns={column_name: new_column_name}, inplace=True)

    harga = df.columns[1:]
    df.loc[:, harga] = df[harga].replace({'-': np.nan, '': np.nan})

    for col in harga:
        df.loc[:, col] = df[col].astype(str).str.replace(',', '', regex=False)
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    yearly_cols = {}
    for col in harga:
        match = re.search(r'\d{4}', col)

        if match:
            year = int(match.group(0)) 
        
            if year not in yearly_cols:
                yearly_cols[year] = []
            yearly_cols[year].append(col)

    kab_hapus = []
    for idx, row in df.iterrows():
        kab_name = row["Kabupaten/Kota"]
        should_be_removed = False

        for year, cols in yearly_cols.items():
            series = row[cols]
            total_days_in_year = len(cols)

            if total_days_in_year == 0:
                continue

            nan_count = series.isna().sum()
            percent_nan_annual = nan_count / total_days_in_year

            if percent_nan_annual > PERCENT_THRESHOLD:
                should_be_removed = True
                break

        if should_be_removed:
            kab_hapus.append(kab_name)

    df = df[~df["Kabupaten/Kota"].isin(kab_hapus)].reset_index(drop=True)
    df[harga] = df[harga].ffill(axis=1)
    df[harga] = df[harga].bfill(axis=1)

    return df


@st.cache_data
def data_preprocessing(df_input):
    df = df_input.copy()
    kabupaten = df.iloc[:, 0]
    numeric_df = df.iloc[:, 1:]

    cleaned_columns = (
        numeric_df.columns
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", "", regex=True)      
            .str.replace(r"[^0-9/]", "", regex=True)  
        )
    
    numeric_df.columns = pd.to_datetime(cleaned_columns, format='%d/%m/%Y', errors="coerce")

    # hitung rata-rata bulanan
    monthly_avg = (
        numeric_df
        .T                                  
        .resample("ME")                     
        .mean()                             
        .T                                  
    )

    monthly_avg.columns = [
        f"{col.strftime('%b%Y')}" for col in monthly_avg.columns
    ]
    monthly_avg.insert(0, "Kabupaten/Kota", kabupaten.values)

    return monthly_avg


@st.cache_data
def data_scaling(df):
    X = df.drop("Kabupaten/Kota", axis=1, errors="ignore")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


@st.cache_data
def get_available_years(df):
    ignore_cols = ["Kabupaten/Kota"]
    years = set()
    datetime_types = (datetime, pd.Timestamp)

    for col in df.columns:
        if isinstance(col, datetime_types):
            col = col.strftime('%d/%m/%Y')

        if col not in ignore_cols:
            # Mencari pola empat digit angka (\d{4}) di akhir string
            match = re.search(r'(\d{4})$', col)
            if match:
                years.add(match.group(1))
    return sorted(list(years))


@st.cache_data
def filter_df_by_years(df, selected_years):
    if not selected_years:
        print("Peringatan: Tidak ada tahun yang dipilih. Mengembalikan DataFrame kosong.")
        return df[['Kabupaten/Kota']].copy()
    
    cols_to_keep = ['Kabupaten/Kota']
    
    for col in df.columns:
        if col not in cols_to_keep:
            for year in selected_years:
                if col.endswith(year):
                    cols_to_keep.append(col)
                    break
    
    df_filtered = df[cols_to_keep].copy()
    return df_filtered


# validasi tahun berurutan
def is_sequential_years(selected_years):
    if not selected_years or len(selected_years) <= 1:
        return True

    int_years = sorted([int(y) for y in selected_years])
    for i in range(len(int_years) - 1):
        if int_years[i+1] - int_years[i] != 1:
            return False
            
    return True


def visualize_cluster(df_result, df_month, sil, dbi, runtime, method_name):
    df_result['Kabupaten/Kota'] = df_result['Kabupaten/Kota'].str.strip()

    st.write("---")
    st.header("📊 Hasil Visualisasi Clustering")

    st.write("### Evaluasi Performa Cluster")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{sil:.4f}")
    with col2:
        st.metric("Davies-Bouldin Index", f"{dbi:.4f}")
    with col3:
        st.metric("Runtime", f"{runtime:.4f}s")

    if sil < 0:
        st.error(f"❌ **Clustering Gagal!** Metode **{method_name}** tidak berhasil membuat *cluster* yang valid.")
        st.info("Coba ubah parameter *clustering* atau gunakan Metode yang berbeda.")
        return
    
    plot_functions = {
        'plot_silhouette_graph': plot_silhouette_graph,
        'plot_pca_scatter': plot_pca_scatter,
        'commodity_correlation': commodity_correlation,
        'plot_commodity_visualizations': plot_commodity_visualizations
    }

    
    st.write("---")
    lottie_placeholder = st.empty()
    display_loading_lottie(lottie_placeholder, height=500, width=500, key="report_loader")

    pdf_bytes = create_report(df_result.copy(), df_month, sil, dbi, runtime, method_name, plot_functions)

    col_map, col_cluster = st.columns(2)
    with col_map:
        show_cluster_map(df_result, "")
    with col_cluster:
        display_cluster_members(df_result, method_name)

        st.write("")
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Laporan_Clustering_{method_name}.pdf",
            mime="application/pdf",
            help="Unduh laporan lengkap",
            on_click="ignore"
        )
    
    lottie_placeholder.empty()
    lottie_placeholder = st.empty()
    display_loading_lottie(lottie_placeholder, height=500, width=500, key="vis_loader")

    X_features = df_result.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore") 
    X_features = data_scaling(X_features)
    labels = df_result["Cluster"].values
    
    fig_sil, avg_sil_score = plot_silhouette_graph(X_features, labels, method_name)
    fig_pca = plot_pca_scatter(X_features, labels, method_name)
    fig_corr = commodity_correlation(df_result)

    col_silhouette, col_pca = st.columns([1, 1])
    with col_silhouette:
        st.write("### 📏 Grafik Silhouette")
        fig_sil.tight_layout()
        st.pyplot(fig_sil)

        buf = io.BytesIO()
        fig_sil.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="📥 Download Plot Silhouette",
            data=buf.getvalue(),
            file_name=f"silhouette_plot_{method_name}.png",
            mime="image/png",
            on_click="ignore"
        )

    with col_pca:
        st.write("### 🖼️ Plot Penyebaran Cluster")
        fig_pca.tight_layout()
        st.pyplot(fig_pca)

        buf = io.BytesIO()
        fig_pca.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="📥 Download Plot PCA",
            data=buf.getvalue(),
            file_name=f"pca_scatter_plot_{method_name}.png",
            mime="image/png",
            on_click="ignore"
        )

    lottie_placeholder.empty()
    lottie_placeholder = st.empty()
    display_loading_lottie(lottie_placeholder, height=500, width=500, key="corr_loader")

    st.write("")
    if fig_corr is not None:
        st.subheader("📐 Matriks Korelasi")
        col_corr, col_spacer1 = st.columns([1,1.1])
        with col_corr:
            fig_corr.tight_layout()
            st.pyplot(fig_corr)

            buf = io.BytesIO()
            fig_corr.savefig(buf, format="png", bbox_inches='tight') 
            buf.seek(0)
            st.download_button(
                label="📥 Download Heatmap Korelasi",
                data=buf.getvalue(),
                file_name="heatmap_korelasi_komoditas.png",
                mime="image/png",
                on_click="ignore"
            )

    lottie_placeholder.empty()

    st.write("### 📉 Analisis Distribusi dan Pergerakan Harga")
    lottie_placeholder = st.empty()
    display_loading_lottie(lottie_placeholder, height=500, width=500, key="plot_loader")

    df_plot = df_result[df_result['Cluster'] != -1]

    all_komoditas_dfs = []

    for prefix, df in df_month.items():
        df_monthly_wide_raw = df.merge(
            df_result[['Kabupaten/Kota', 'Cluster']], 
            on='Kabupaten/Kota', 
            how='inner'
        )

        id_cols = ['Kabupaten/Kota', 'Cluster']
        value_cols = [col for col in df_monthly_wide_raw.columns if col not in id_cols]
        
        df_long = df_monthly_wide_raw.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='Periode_Raw',
            value_name='Harga'
        )

        df_long['BulanTahun'] = df_long['Periode_Raw'].str.replace(f"{prefix}_", "", regex=False)
        all_komoditas_dfs.append(df_long)

    if not all_komoditas_dfs:
        st.error("Data bulanan komoditas tidak ditemukan.")
        return   

    df_all_long = pd.concat(all_komoditas_dfs, ignore_index=True)
    df_grouped_all = df_all_long.groupby(['BulanTahun', 'Cluster'])['Harga'].mean().reset_index()

    df_combined_wide = df_grouped_all.pivot_table(
        index='Cluster', 
        columns='BulanTahun', 
        values='Harga'
    ).reset_index()

    df_combined_wide['Kabupaten/Kota'] = 'Rata-Rata Semua'
    df_combined_wide = df_combined_wide[['Kabupaten/Kota', 'Cluster'] + [col for col in df_combined_wide.columns if col not in ['Kabupaten/Kota', 'Cluster']]]

    all_komoditas = sorted(
        set("_".join(col.split("_")[:-1]) 
            for col in df_plot.columns 
            if "_" in col and col != "Cluster"
        )
    )

    if len(all_komoditas) > 1:
        tabs_prefixes = ['ALL_KOMODITAS'] + all_komoditas
        tabs_display = ['(RATA-RATA)'] + [p.replace('_', ' ').upper() for p in all_komoditas]
        
    else:
        tabs_prefixes = all_komoditas
        tabs_display = [p.replace('_', ' ').upper() for p in all_komoditas]\
    
    tabs = st.tabs(tabs_display)
    lottie_placeholder.empty()

    for i, prefix in enumerate(tabs_prefixes):
        display_name = tabs_display[i]
        
        if prefix == 'ALL_KOMODITAS':
            df_monthly_wide_raw_current = df_combined_wide
        else:
            df_monthly_raw_single = df_month.get(prefix)
            if df_monthly_raw_single is None:
                st.warning(f"Data bulanan harga untuk {display_name} tidak ditemukan di map 'df_month'.")
                continue
            
            df_monthly_wide_raw_current = df_monthly_raw_single.merge(
                df_result[['Kabupaten/Kota', 'Cluster']], 
                on='Kabupaten/Kota', 
                how='inner'
            )

        with tabs[i]:
            lottie_placeholder = st.empty()
            display_loading_lottie(lottie_placeholder, height=500, width=500, key=f"tab{i}_loader")
            fig_box, fig_line = plot_commodity_visualizations(df_monthly_wide_raw_current, prefix)
            
            col_box, col_line = st.columns([0.6, 1])
            with col_box:
                st.write(f"##### Distribusi Harga Komoditas {display_name} Tahunan")
                if fig_box:
                    fig_box.tight_layout()
                    st.pyplot(fig_box)
                    
                    buf = io.BytesIO()
                    fig_box.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label=f"📥 Download Boxplot",
                        data=buf.getvalue(),
                        file_name=f"boxplot_{prefix.lower()}.png",
                        mime="image/png",
                        key=f"dl_box_{prefix}",
                        on_click="ignore"
                    )
                else:
                    st.warning("Visualisasi Boxplot tidak dapat dibuat.")

            with col_line:
                st.write(f"##### Pergerakan Harga {display_name}")
                if fig_line:
                    fig_line.tight_layout()
                    st.pyplot(fig_line)

                    buf = io.BytesIO()
                    fig_line.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label=f"📥 Download Lineplot",
                        data=buf.getvalue(),
                        file_name=f"lineplot_{prefix.lower()}.png",
                        mime="image/png",
                        key=f"dl_line_{prefix}",
                        on_click="ignore"
                    )
                else:
                    st.info("Visualisasi Lineplot tidak dapat dibuat (kemungkinan karena data bulanan yang kurang).")
            lottie_placeholder.empty()
    plt.close('all')


def plot_silhouette_graph(X, labels, method_name):
    if method_name.upper() == "DBSCAN":
        mask = labels != -1
        X_data = X[mask]
        labels_data = labels[mask]
        title_suffix = " (tanpa noise)"
    else: 
        X_data = X
        labels_data = labels
        title_suffix = ""

    unique_labels = np.unique(labels_data)
    if len(unique_labels) <= 1:
        st.warning(f"‼️ Tidak cukup klaster yang valid (> 1 klaster) untuk membuat Silhouette Plot pada metode {method_name}.")
        return
    
    try:
        silhouette_vals = silhouette_samples(X_data, labels_data)
        cluster_labels = unique_labels
        
        fig_silhouette, ax_silhouette = plt.subplots(figsize=(6, 4))
        y_lower = 10
        
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[labels_data == c]
            c_silhouette_vals.sort()
            
            y_upper = y_lower + len(c_silhouette_vals)
            
            ax_silhouette.fill_betweenx(np.arange(y_lower, y_upper),
                                        0, c_silhouette_vals, 
                                        alpha=0.7)
            
            ax_silhouette.text(-0.05, y_lower + 0.5 * len(c_silhouette_vals), str(c))
            y_lower = y_upper + 10 
            
        # Garis Rata-rata Silhouette
        avg_silhouette_score = np.mean(silhouette_vals)
        ax_silhouette.axvline(x=avg_silhouette_score, color="red", linestyle="--")
        
        ax_silhouette.set_title(f"Silhouette Plot {method_name}{title_suffix}", fontsize=14)
        ax_silhouette.set_xlabel("Nilai Silhouette", fontsize=12)
        ax_silhouette.set_ylabel("Klaster", fontsize=12)
        ax_silhouette.set_yticks([]) 
        ax_silhouette.set_xlim([-0.1, 1.0]) 
        plt.close()

        return fig_silhouette, avg_silhouette_score

    except ValueError as e:
        st.error(f"⚠️ Gagal membuat Silhouette Plot. Error: {e}")
        st.info("Pastikan jumlah sampel (data) lebih besar dari jumlah klaster yang dipilih.")


def plot_pca_scatter(X, labels, method_name):
    
    n_samples, n_features = X.shape
    if n_features < 2:
        st.warning(f"⚠️ Data hanya memiliki {n_features} fitur. PCA 2D membutuhkan minimal 2 fitur.")
        return
    if n_samples < 2:
        st.warning(f"⚠️ Data hanya memiliki {n_samples} sampel. PCA tidak dapat dilakukan.")
        return

    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        
        sns.scatterplot(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            hue=labels, 
            palette="tab10", 
            s=50, 
            legend="full", 
            ax=ax_pca
        )

        ax_pca.set_title(f"Visualisasi Hasil Clustering {method_name} (PCA 2D)", fontsize=14)
        ax_pca.set_xlabel(f"PCA 1 (Explained Variance: {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        ax_pca.set_ylabel(f"PCA 2 (Explained Variance: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
        
        ax_pca.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.close()
        return fig_pca
        
    except Exception as e:
        st.error(f"⚠️ Gagal membuat PCA Scatter Plot. Error: {e}")


def plot_commodity_visualizations(df_monthly_wide_raw, commodity):
    df_monthly_wide_raw = df_monthly_wide_raw[df_monthly_wide_raw['Cluster'] != -1].copy()
    display_name = commodity.replace('_', ' ').upper()
    id_cols = ['Kabupaten/Kota', 'Cluster']
    value_cols = [col for col in df_monthly_wide_raw.columns if col not in id_cols]

    if not value_cols:
        st.warning("Data bulanan harga tidak ditemukan di DataFrame gabungan.")
    
    df_melt = df_monthly_wide_raw.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name='Periode',
                value_name='Harga'
            )
    
    if commodity != "ALL_KOMODITAS":
        df_melt["Tanggal"] = (
            df_melt["Periode"]
                .str.replace(f"{commodity}_", "", regex=False)
                .apply(lambda x: pd.to_datetime(x, format="%b%Y", errors="coerce"))
        )
        df_melt['Tahun'] = df_melt['Tanggal'].dt.year.astype(str)

    else:
        df_melt["Tanggal"] = df_melt["Periode"].apply(lambda x: pd.to_datetime(x, format="%b%Y", errors="coerce"))
        df_melt['Tahun'] = df_melt['Tanggal'].dt.year.astype(str)

    df_boxplot = df_melt.dropna(subset=['Harga', 'Tahun']).copy()
    fig_box_tahunan, ax_box_tahunan = plt.subplots(figsize=(6, 5))
    tahun_order = sorted(df_boxplot['Tahun'].unique())
    
    sns.boxplot(
        data=df_boxplot, 
        x="Tahun", 
        y="Harga", 
        hue="Cluster", 
        palette="Set1", 
        ax=ax_box_tahunan,
        order=tahun_order
    )
    
    box_title = f"Boxplot Distribusi Harga Tahunan ({display_name})"
    ax_box_tahunan.set_title(box_title, fontsize=14)
    ax_box_tahunan.set_xlabel("Tahun", fontsize=12)
    ax_box_tahunan.set_ylabel("Harga", fontsize=12)
    ax_box_tahunan.grid(True, alpha=0.3)
    ax_box_tahunan.legend(title='Cluster', loc='upper right')

    df_line_grouped = df_melt.dropna(subset=['Tanggal']).groupby(['Tanggal', 'Cluster'])['Harga'].mean().reset_index()

    fig_line, ax_line = plt.subplots(figsize=(10, 5))
        
    sns.lineplot(
        data=df_line_grouped.sort_values('Tanggal'),
        x='Tanggal',
        y='Harga',
        hue='Cluster',
        palette="Set1",
        ax=ax_line,
        # marker='o', 
        errorbar=None 
    )
    
    line_title = f'Pergerakan Harga Bulanan Rata-Rata ({display_name})'
    ax_line.set_title(line_title, fontsize=14)
    ax_line.set_xlabel('Bulan', fontsize=12)
    ax_line.set_ylabel('Harga Rata-Rata Bulanan', fontsize=12)
    
    date_form = mdates.DateFormatter("%b %Y")
    ax_line.xaxis.set_major_formatter(date_form)
    ax_line.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Tampilkan setiap 3 bulan
    
    ax_line.tick_params(axis='x', rotation=45)
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(title='ID Cluster', loc='upper left', bbox_to_anchor=(1, 1))
    plt.close()

    return fig_box_tahunan, fig_line


def show_cluster_map(df_result, filename_path):

    df_map = pd.read_csv('../misc/mapping.csv')
    df_merged = df_result.merge(df_map, on="Kabupaten/Kota", how="inner")

    if {"Latitude", "Longitude"}.issubset(df_merged.columns):
        avg_lat = df_merged["Latitude"].mean()
        avg_lon = df_merged["Longitude"].mean()

        m = folium.Map(
            location=[avg_lat, avg_lon+8], 
            zoom_start=5.2,
        )

        unique_clusters = sorted(df_merged["Cluster"].unique())
        colors = sns.color_palette("tab10", n_colors=len(unique_clusters)).as_hex()
        color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        color_map[-1] = "gray"  # warna untuk noise / outlier

        legend_html = ''
        legend_html += '''
             <div style="position: fixed; 
                         bottom: 20px; left: 20px; width: 150px; height: auto; 
                         border:2px solid gray; z-index:9999; font-size:12px;
                         background-color: white; opacity: 0.9;">
               &nbsp; <b>Legend Cluster</b> <br>
        '''

        for cluster_id in unique_clusters:
            color = color_map.get(cluster_id)
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise / Outlier (-1)'
            
            legend_html += f'''
               &nbsp; <i style="background:{color}; color:{color}; 
                               width: 10px; height: 10px; display: inline-block; 
                               border-radius: 50%;"></i> 
               &nbsp; {label} <br>
            '''

        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

        for _, row in df_merged.iterrows():
            cluster_id = int(row["Cluster"])
            color = color_map.get(cluster_id, "gray")

            popup_html = f"<b>{row['Kabupaten/Kota']}</b><br>Cluster: {cluster_id}"
        
            popup_obj = folium.Popup(
                html=popup_html,
                max_width=300, 
                min_width=100  
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

        if filename_path != "":
            try:
                
                temp_html_path = tempfile.mktemp(suffix=".html")
                m.save(temp_html_path)

                chrome_options = Options()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')

                driver = webdriver.Chrome(options=chrome_options)
                DRIVER_WIDTH = 870
                DRIVER_HEIGHT = 650 
                driver.set_window_size(DRIVER_WIDTH, DRIVER_HEIGHT)
                driver.get(f'file://{os.path.abspath(temp_html_path)}')

                time.sleep(1)
                driver.save_screenshot(filename_path)
                driver.quit()

                START_X = 0 
                WIDTH = 850   
                HEIGHT = 500

                img = Image.open(filename_path)
                img_cropped = img.crop((START_X, 0, START_X + WIDTH, HEIGHT))
                img_cropped.save(filename_path)

                os.unlink(temp_html_path)

                return True
            except Exception as e:
                print(f"Gagal menyimpan peta sebagai HTML: {e}")

        st.write("### 🗺️ Peta Sebaran Cluster")
        st_folium(
            m, 
            width=850, 
            height=500, 
            key='folium_cluster_map_stable', 
            returned_objects=[] 
        )

    else:
        print("test")
        st.warning("Kolom Latitude dan Longitude belum tersedia pada data hasil penggabungan.")            


def display_cluster_members(df_result, method_name):
    st.subheader(f"👥 Anggota Klaster Hasil {method_name}")
    
    if 'Cluster' not in df_result.columns:
        st.warning("Kolom 'Cluster' tidak ditemukan. Pastikan proses clustering sudah berhasil.")
        return

    cluster_groups = df_result.groupby('Cluster')['Kabupaten/Kota'].apply(list).reset_index()
    num_clusters = cluster_groups.query('Cluster != -1')['Cluster'].nunique()
    st.info(f"Ditemukan total **{num_clusters}** klaster. Klik untuk melihat daftar anggota.")


    for index, row in cluster_groups.iterrows():
        cluster_id = row['Cluster']
        members = row['Kabupaten/Kota']
        num_members = len(members)

        members = row['Kabupaten/Kota']
        member_list = "\n".join([f"- {member}" for member in members])

        if(cluster_id == -1):
            with st.expander(f"Noise/Outlier ({num_members} Anggota)"):
                # st.write(pd.DataFrame(members, columns=["Kabupaten/Kota"]))
                st.markdown(member_list)
                st.caption(f"Total anggota: {num_members}")
        else:
            with st.expander(f"Klaster {cluster_id} ({num_members} Anggota)"):
                # st.write(pd.DataFrame(members, columns=["Kabupaten/Kota"]))
                st.markdown(member_list)
                st.caption(f"Total anggota: {num_members}")


def validate_uploaded_data(df_sheets):
    DATE_REGEX = re.compile(r'^\s*\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}\s*$')
    DATE_FORMAT = '%d/%m/%Y'
    datetime_types = (datetime, pd.Timestamp)

    reference_periods = None 
    first_sheet_name = None

    try:
        for sheet_name, df in df_sheets.items():
            if df.empty:
                return False, f"Sheet '{sheet_name}' kosong, mohon isi datanya."
            
            # dta harga kosong semua
            data_harga = df.iloc[:, 1:]
            if data_harga.isna().all().all():
                raise Exception(f"Sheet '{sheet_name}' tampaknya **kosong** mulai dari kolom kedua. "
                            f"Mohon masukkan data harga untuk analisis.")
        
            if (data_harga.astype(str).replace(r'^\s*$', np.nan, regex=True).isna().all().all()):
                raise Exception(f"Sheet '{sheet_name}' tampaknya **kosong** atau hanya berisi spasi "
                                f"mulai dari kolom kedua. Mohon masukkan data harga yang valid.")
                
            data_str = data_harga.astype(str)

            data_str_cleaned = (
                data_str
                .replace(r'^\s*-\s*$', '', regex=True)  # ubah "-" jadi kosong
                .replace({',': ''}, regex=False)
                .replace(r'\s+', '', regex=True)        # hapus spasi berlebih
            )

            data_numeric_check = data_str_cleaned.apply(pd.to_numeric, errors='coerce')
            # konversi gagal, jadi NaN
            non_numeric_mask = data_numeric_check.isna() & (data_str_cleaned != '')

            error_positions = []
            for i, j in zip(*np.where(non_numeric_mask)):
                val = data_str_cleaned.iloc[i, j]
                # Abaikan nilai kosong atau NaN
                if val in ['', 'nan', 'NaN', None]:
                    continue
                # Cek apakah ada huruf alfabet di kolom harga
                if bool(re.search(r'[A-Za-z]', val)):
                    error_positions.append((i, j, val))

            if error_positions:
                # Ambil error pertama (atau bisa juga dibuat daftar semua)
                row_index = error_positions[0][0] + 1
                col_name = data_harga.columns[error_positions[0][1]]
                original_value = error_positions[0][2]

                raise Exception(
                    f"Sheet '{sheet_name}': Nilai pada kolom '{col_name}' di baris {row_index} "
                    f"('{original_value}') mengandung huruf dan tidak valid sebagai angka. "
                    f"Pastikan nilainya numerik saja (contoh: 10000 atau 10,000)."
                )

            first_col_name = df.columns[0]
            if df[first_col_name].dtype != 'object':
                try:
                    df[first_col_name] = df[first_col_name].astype(str)
                except Exception as e:
                    raise Exception("Pastikan kolom pertama adalah teks!")
            
            harga_cols = df.columns[1:]
            
            if len(harga_cols) == 0:
                raise Exception(f"Sheet '{sheet_name}': Data tidak memiliki kolom tanggal harga/jumlah.")

            new_cols = []
            for col in harga_cols:
                if isinstance(col, datetime_types):
                    try:
                        col_str = col.strftime(DATE_FORMAT)
                    except Exception as e:
                        raise Exception(f"Sheet '{sheet_name}': Nama Kolom '{col}' tidak sesuai format **dd/mm/yyyy** (cth: 01/01/2020).")
                    new_cols.append(col)

                elif isinstance(col, str):
                    col_str = col.replace(" ", "")
                    if not DATE_REGEX.match(col_str):
                        st.write("NI")
                        raise Exception(f"Sheet '{sheet_name}': Nama kolom '{col_str}' tidak sesuai format **dd/mm/yyyy** (cth: 01/01/2020).")

                    col_dt = datetime.strptime(col_str, DATE_FORMAT)
                    new_cols.append(col_dt)
                
            df.columns = [first_col_name] + new_cols
            current_periods = sorted(pd.to_datetime(new_cols).tolist())

            month_periods = sorted(set(pd.to_datetime(new_cols).to_period('M')))
            if len(month_periods) < 3:
                raise Exception(
                    f"Sheet '{sheet_name}': Data hanya memiliki {len(month_periods)} bulan ({[str(m) for m in month_periods]}). Minimal diperlukan 3 bulan data untuk analisis."
                )      

            # validasi periode antar sheets
            if reference_periods is None:
                reference_periods = current_periods
                first_sheet_name = sheet_name
            elif current_periods != reference_periods:
                ref_set = set(reference_periods)
                current_set = set(current_periods)

                if ref_set != current_set:
                    missing = ref_set - current_set
                    extra = current_set - ref_set
                    
                    error_msg = f"Periode tanggal pada Sheet '{sheet_name}' tidak sama dengan Sheet '{first_sheet_name}'. "
                    if missing:
                        error_msg += f"Tanggal yang HILANG di '{sheet_name}': {sorted([d.strftime(DATE_FORMAT) for d in missing])}. "
                    if extra:
                        error_msg += f"Tanggal yang BERLEBIH di '{sheet_name}': {sorted([d.strftime(DATE_FORMAT) for d in extra])}."
                    
                    raise Exception(error_msg)

        return True, "Validasi berhasil!"
    except Exception as e:
        return False, e
    

def commodity_correlation(df):
    harga_cols = df.columns[1:]

    KOMODITAS_PATTERN = r'^(.*?)_[A-Za-z]{3}\d{4}$'
    get_komoditas_name = lambda col: re.match(KOMODITAS_PATTERN, col).group(1) if re.match(KOMODITAS_PATTERN, col) else None
    
    komoditas_list = sorted(
        set(get_komoditas_name(col) for col in harga_cols if get_komoditas_name(col) is not None)
    )

    # st.write(komoditas_list)
    if len(komoditas_list) > 1:
        df_komoditas_mean = df[['Kabupaten/Kota']].copy()
        for komoditas in komoditas_list:
            kolom_komoditas = [col for col in harga_cols if get_komoditas_name(col) == komoditas]
            df_komoditas_mean[komoditas] = df[kolom_komoditas].mean(axis=1)

        corr_matrix = df_komoditas_mean.drop(columns=['Kabupaten/Kota']).corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f",
            linewidths=.5,
            linecolor='black',
            ax=ax
        )
        ax.set_title("Heatmap Korelasi Antar Komoditas Pangan (Rata-Rata Harga)", fontsize=16)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        return fig
    else:
        return None


def create_report(df_result, df_month, sil, dbi, runtime, method_name, plot_functions):
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    MIN_HEIGHT_REQUIRED = 87

    plot_silhouette_graph = plot_functions['plot_silhouette_graph']
    plot_pca_scatter = plot_functions['plot_pca_scatter']
    commodity_correlation = plot_functions['commodity_correlation']
    plot_commodity_visualizations = plot_functions['plot_commodity_visualizations']

    no = 1
    INDENT_X = 15

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Laporan Hasil Clustering Komoditas', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 5, f'Metode Clustering: {method_name.upper()}', 0, 1, 'C')
    pdf.ln(5)

    # Metrik Kinerja
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 7, f'{no}. Metrik Kinerja Cluster', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.set_x(INDENT_X)
    pdf.cell(60, 5, 'Silhouette Score:', 0, 0, 'L')
    pdf.cell(0, 5, f'{sil:.4f}', 0, 1, 'L')
    pdf.set_x(INDENT_X)
    pdf.cell(60, 5, 'Davies-Bouldin Index:', 0, 0, 'L')
    pdf.cell(0, 5, f'{dbi:.4f}', 0, 1, 'L')
    pdf.set_x(INDENT_X)
    pdf.cell(60, 5, 'Waktu Eksekusi (Runtime):', 0, 0, 'L')
    pdf.cell(0, 5, f'{runtime:.4f} detik', 0, 1, 'L')
    pdf.ln(3)
    no+=1

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 7, f'{no}. Anggota Cluster', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)

    grouped_clusters = df_result.groupby('Cluster')['Kabupaten/Kota'].apply(list).to_dict()

    valid_clusters = sorted([c for c in grouped_clusters.keys() if c != -1])
    noise_cluster = grouped_clusters.get(-1)

    for cluster_id in valid_clusters:
        kab_list = grouped_clusters[cluster_id]
        
        pdf.set_font('Arial', 'BU', 10)
        pdf.set_x(INDENT_X)
        pdf.cell(0, 6, f'Cluster {cluster_id} ({len(kab_list)} Anggota):', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        num_items = len(kab_list)
        mid_point = (num_items + 1) // 2
        col1 = kab_list[:mid_point]
        col2 = kab_list[mid_point:]
        max_len = max(len(col1), len(col2))
        
        for i in range(max_len):
            kab1 = col1[i] if i < len(col1) else ""
            kab2 = col2[i] if i < len(col2) else ""
            
            pdf.set_x(INDENT_X)
            pdf.cell(90, 5, f'- {kab1}', 0, 0, 'L') 
            
            if kab2 == "":
                pdf.ln(4)
                break
            else:
                pdf.cell(90, 5, f'- {kab2}', 0, 1, 'L')

        pdf.ln(2)

    if noise_cluster:
        kab_list_noise = noise_cluster 
    
        pdf.set_font('Arial', 'BI', 10) 
        pdf.set_x(INDENT_X)
        pdf.cell(0, 6, f'Noise/Outlier ({len(kab_list_noise)} Anggota):', 0, 1, 'L')
        pdf.set_font('Arial', '', 10) 
        
        num_items = len(kab_list_noise)
        mid_point = (num_items + 1) // 2
        col1 = kab_list_noise[:mid_point]
        col2 = kab_list_noise[mid_point:]
        max_len = max(len(col1), len(col2))
        
        for i in range(max_len):
            kab1 = col1[i] if i < len(col1) else ""
            kab2 = col2[i] if i < len(col2) else ""
            
            pdf.set_x(INDENT_X)
            pdf.cell(90, 5, f'- {kab1}', 0, 0, 'L') 
            
            if kab2 == "":
                pdf.ln(1)
                break 
            else:
                pdf.cell(90, 5, f'- {kab2}', 0, 1, 'L')

        pdf.ln(3)
    else:
        pdf.ln(2)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 7, f'{no}. Peta Sebaran Cluster', 0, 1, 'L') 
    
    temp_filename_map = tempfile.mktemp(suffix=".png")
    map_saved_successfully = show_cluster_map(df_result, temp_filename_map)

    if map_saved_successfully:
        pdf.image(temp_filename_map, x=15, y=pdf.get_y(), w=180, type='PNG') 
        os.unlink(temp_filename_map)
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, 'Peta tidak dapat dihasilkan karena data kurang atau gagal konversi ke gambar.', 0, 1, 'L')
        
    pdf.ln(115) 
    no += 1

    X_features = df_result.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore") 
    labels = df_result["Cluster"].values
    
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_features_scaled = scaler.fit_transform(X_features)
    except:
         X_features_scaled = X_features.values
    
    
    fig_sil, _ = plot_silhouette_graph(X_features_scaled, labels, method_name)
    fig_pca = plot_pca_scatter(X_features_scaled, labels, method_name)

    # Silhouette Plot & Scatter Plot
    pdf.set_font('Arial', 'B', 12)
    pdf.set_xy(10, pdf.get_y())
    pdf.cell(0, 7, f'{no}. Plot Silhouette & Scatter Plot', 0, 1, 'L')
    buf_sil = io.BytesIO()
    fig_sil.savefig(buf_sil, format="png", bbox_inches='tight')
    plot_y = pdf.get_y()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(buf_sil.read())
        temp_filename_sil = tmp.name

    pdf.image(buf_sil, x=10, y=plot_y, w=80, type='PNG')
    os.unlink(temp_filename_sil)
    
    buf_pca = io.BytesIO()
    fig_pca.savefig(buf_pca, format="png", bbox_inches='tight')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(buf_pca.read())
        temp_filename_pca = tmp.name

    pdf.image(buf_pca, x=95, y=plot_y, w=95, type='PNG')
    os.unlink(temp_filename_pca)
    no += 1
    pdf.ln(80)
    
    # Matriks Korelasi
    fig_corr = commodity_correlation(df_result)
    if fig_corr is not None:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 7, f'{no}. Matriks Korelasi', 0, 1, 'L')
        buf_corr = io.BytesIO()
        fig_corr.savefig(buf_corr, format="png", bbox_inches='tight')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf_corr.read())
            temp_filename_corr = tmp.name

        pdf.image(buf_corr, x=10, y=pdf.get_y(), w=120, type='PNG')
        pdf.ln(110)
        os.unlink(temp_filename_corr)
        no += 1
            
    if pdf.get_y() + MIN_HEIGHT_REQUIRED > (pdf.h - pdf.b_margin):
        pdf.add_page()

    # Visualisasi Harga per Komoditas
    df_plot = df_result[df_result['Cluster'] != -1]
    all_komoditas = sorted(
        set("_".join(col.split("_")[:-1]) 
            for col in df_plot.columns 
            if "_" in col and col != "Cluster"
        )
    )
    
    tabs_prefixes = all_komoditas
    tabs_display = [p.replace('_', ' ').upper() for p in all_komoditas]

    if len(all_komoditas) > 1:
        tabs_prefixes.insert(0, 'ALL_KOMODITAS')
        tabs_display.insert(0, 'SEMUA KOMODITAS (RATA-RATA)')

    df_all_long = pd.concat([
        df_month.get(prefix).merge(
            df_result[['Kabupaten/Kota', 'Cluster']], 
            on='Kabupaten/Kota', 
            how='inner'
        ).melt(
            id_vars=['Kabupaten/Kota', 'Cluster'], value_vars=[col for col in df_month.get(prefix).columns if col != 'Kabupaten/Kota'],
            var_name='Periode_Raw', value_name='Harga'
        ).assign(BulanTahun=lambda x: x['Periode_Raw'].str.replace(f"{prefix}_", "", regex=False))
        for prefix in df_month.keys() if df_month.get(prefix) is not None
    ], ignore_index=True)

    df_grouped_all = df_all_long.groupby(['BulanTahun', 'Cluster'])['Harga'].mean().reset_index()
    df_combined_wide = df_grouped_all.pivot_table(
        index='Cluster', columns='BulanTahun', values='Harga'
    ).reset_index().assign(**{'Kabupaten/Kota': 'Rata-Rata Semua'})
    df_combined_wide = df_combined_wide[['Kabupaten/Kota', 'Cluster'] + [col for col in df_combined_wide.columns if col not in ['Kabupaten/Kota', 'Cluster']]]
    
    PLOT_WIDTH = 75
    MARGIN_LEFT = 10
    MARGIN_MID = MARGIN_LEFT + PLOT_WIDTH + 5

    pdf.cell(0, 10, f'{no}. Analisis Harga Komoditas', 0, 1, 'L')

    for i, prefix in enumerate(tabs_prefixes):

        if pdf.get_y() + MIN_HEIGHT_REQUIRED > (pdf.h - pdf.b_margin):
            pdf.add_page()

        display_name = tabs_display[i]
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, f'{display_name}', 0, 1, 'L')
        
        if prefix == 'ALL_KOMODITAS':
            df_monthly_wide_raw_current = df_combined_wide
        else:
            df_monthly_raw_single = df_month.get(prefix)
            if df_monthly_raw_single is None: continue
            df_monthly_wide_raw_current = df_monthly_raw_single.merge(
                df_result[['Kabupaten/Kota', 'Cluster']], on='Kabupaten/Kota', how='inner'
            )

        fig_box, fig_line = plot_commodity_visualizations(df_monthly_wide_raw_current, prefix)
        start_y = pdf.get_y()
        max_height = 0 

        pdf.set_font('Arial', 'B', 10)
        pdf.set_x(MARGIN_LEFT)
        pdf.cell(PLOT_WIDTH, 5, 'a. Distribusi Harga Tahunan', 0, 0, 'L')
        pdf.set_y(start_y + 5) 

        buf_box = io.BytesIO()
        if fig_box:
            fig_box.savefig(buf_box, format="png", bbox_inches='tight')
            buf_box.seek(0)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(buf_box.read())
                temp_filename_box = tmp.name

            pdf.image(temp_filename_box, x=MARGIN_LEFT, y=pdf.get_y(), w=PLOT_WIDTH)
            max_height = max(max_height, 90) 
            os.unlink(temp_filename_box) 
            
        else:
            pdf.set_font('Arial', 'I', 10)
            pdf.set_y(pdf.get_y())
            pdf.set_x(MARGIN_LEFT)
            pdf.cell(PLOT_WIDTH, 5, 'Tidak dapat membuat Boxplot.', 0, 1, 'L')
            max_height = max(max_height, 10)

        pdf.set_y(start_y)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_x(MARGIN_MID)
        pdf.cell(PLOT_WIDTH, 5, 'b. Pergerakan Harga Bulanan Rata-Rata', 0, 1, 'L')

        buf_line = io.BytesIO()
        if fig_line:
            fig_line.savefig(buf_line, format="png", bbox_inches='tight')
            buf_line.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(buf_line.read())
                temp_filename_line = tmp.name

            pdf.image(temp_filename_line, x=MARGIN_MID, y=pdf.get_y(), w=PLOT_WIDTH+25)
            max_height = max(max_height, 90)
            os.unlink(temp_filename_line) 

        else:
            pdf.set_font('Arial', 'I', 10)
            pdf.set_y(pdf.get_y())
            pdf.set_x(MARGIN_MID)
            pdf.cell(PLOT_WIDTH, 5, 'Tidak dapat membuat Lineplot.', 0, 1, 'L')
            max_height = max(max_height, 10)    

        pdf.set_y(start_y + max_height - 20)

    return bytes(pdf.output(dest='S'))


# def hash_dataframe(df):
#     df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
#     df_string = df_sorted.to_json() 
#     return hashlib.sha256(df_string.encode('utf-8')).hexdigest()


def standardized_columns(df):
    converted_df = {}
    DATE_FORMAT = '%d/%m/%Y'
    datetime_types = (datetime, pd.Timestamp)

    for name, df in df.items():
        df_temp = df.copy()
        cols_to_process = df_temp.columns[1:]
        new_columns = [df_temp.columns[0]]

        for col in cols_to_process:
            if isinstance(col, datetime_types):
                formatted_name = col.strftime(DATE_FORMAT)
                new_columns.append(formatted_name)
            elif isinstance(col, str):
                new_columns.append(col.strip())
        
        df_temp.columns = new_columns
        df_temp.columns = list(map(str.strip, df_temp.columns))
        converted_df[name] = df_temp
    
    return converted_df


def reset_uploaded_data():
    st.session_state.df = None
    st.session_state.isDefault = False
    # st.session_state.uploader_key += 1






def app():

# --- Inisialisasi session_state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "isDefault" not in st.session_state:
        st.session_state.isDefault = False
    if "cluster_executed" not in st.session_state:
        st.session_state.cluster_executed = False
    if "current_method" not in st.session_state:
        st.session_state.current_method = "K-Means"
    if "final_df_input" not in st.session_state:
        st.session_state.final_df_input = None
    if "final_k" not in st.session_state:
        st.session_state.final_k = None
    if "final_eps" not in st.session_state:
        st.session_state.final_eps = None
    if "final_minpts" not in st.session_state:
        st.session_state.final_minpts = None
    if 'kmeans_preview' not in st.session_state:
        st.session_state.kmeans_preview = None
    if 'dbscan_preview' not in st.session_state:
        st.session_state.dbscan_preview = None
    if 'df_input_hash' not in st.session_state:
        st.session_state.df_input_hash = ""
    if 'last_processed_keys' not in st.session_state:
        st.session_state.last_processed_keys = []
    if 'last_processed_years' not in st.session_state:
        st.session_state.last_processed_years = []        
    if 'show_uploader' not in st.session_state:
        st.session_state.show_uploader = True
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key   = 0

    st.title("📈 Dashboard Analisis Harga Pangan")
    st.write("---")

    st.markdown("#### Upload Dataset")
    st.info("""
        1. Wajib menggunakan file dengan **.xlsx** format.
        2. Default dataset dan template dataset dapat diunduh pada tombol berikut.
        3. Dataset yang diinput merupakan data harian harga komoditas pangan.
        4. Format nominal harga pangan tanpa separator seperti '10000' dan '10500'.
        5. Formmat kolom tanggal adalah 'dd/MM/yyyy'.
        6. Jumlah komoditas ditentukan berdasarkan sheet yang ada dalam file yang di-upload.
        7. Penamaan sheet harus dalam huruf kecil dengan spasi ditandai oleh underscore seperti 'bawang_merah',
        8. Pastikan periode harian harga pangan untuk seluruh komoditas adalah sama.
        9. Pastikan dataset sudah sesuai dengan format dan ketentuan.
    """)
    


    
    df = None
    uploaded_file = None

    col_template, col_default, _ = st.columns([0.35, 0.35, 1.2]) 
    try:
        with col_template:
            with open(TEMPLATE_FILE_PATH, "rb") as file:
                st.download_button(
                    label="📥 Download Template (.xlsx)",
                    data=file,
                    file_name="template_data_pangan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ) 

        with col_default:
            if os.path.exists(READY_FILE_PATH):
                    if st.button("🚀 Default Dataset"):
                        df_default = pd.read_excel(READY_FILE_PATH, sheet_name=None)
                        st.session_state.uploader_key += 1

                        st.session_state.df = df_default
                        st.session_state.isDefault = True
                        st.toast("✅ Dataset default berhasil dimuat!")
                # else:
                #     st.session_state.df = None


    except FileNotFoundError:
        traceback.print_exc()
        st.error(f"⚠️ Gagal memuat file!")
    
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)", 
        type=["xlsx"],
        key=f"xlsx_uploader_key_{st.session_state.uploader_key}",
        on_change=reset_uploaded_data,
        accept_multiple_files=False
        )
    
    if uploaded_file is not None:
        df_uploaded = pd.read_excel(uploaded_file, sheet_name=None)
        is_valid, validation_message = validate_uploaded_data(df_uploaded)

        if is_valid:
            st.session_state.df = df_uploaded
            st.session_state.isDefault = False
            st.toast("✅ Dataset berhasil diunggah!")
        else:
            st.error(f"⚠️ **Gagal Validasi Data!** {validation_message}")
            st.session_state.df = None

    df = st.session_state.df
    if df is not None:
                
        try:
            st.write("---")
            # standarisasi semua kolom tanggal
            df = standardized_columns(df)

            sheet_names = list(df.keys())
            display_options = []
            name_map = {}

            with st.spinner("Membaca dataset..."):
                for name in sheet_names:
                    formatted_name = name.replace('_', ' ').upper()
                    display_options.append(formatted_name)
                    name_map[formatted_name] = name

            first_sheet_name = list(df.keys())[0]
            available_years = get_available_years(df[first_sheet_name])

            col_sheets, col_year = st.columns(2)

            with col_sheets:
                selected_keys = []
                selected_display = []
                lottie2_placeholder = st.empty()

                # hanya satu komoditas gak usah pilih
                if len(sheet_names) == 1:
                    selected_keys = sheet_names
                    selected_display = display_options
                    all_sheets = True
                    display_lottie_bot(lottie2_placeholder, position="left", height=197, width=300, key="blink2")
                
                else:
                    st.markdown("")
                    all_sheets = st.checkbox("SEMUA KOMODITAS", False)
                    if all_sheets:
                        selected_keys = sheet_names
                        selected_display = display_options
                    else:
                        selected_commodity = st.multiselect(
                            "Atau Pilih Komoditas", 
                            options = display_options,
                            placeholder = "Pilih komoditas...")
                        
                        if selected_commodity:
                            for formatted_name in selected_commodity:
                                key = name_map[formatted_name]
                                selected_keys.append(key)
                                selected_display.append(formatted_name)
                        elif len(selected_commodity) == 0:
                            st.info("Silakan pilih komoditas untuk memulai pemrosesan data.")

            with col_year:
                # first_sheet_name = list(df.keys())[0]
                # available_years = get_available_years(df[first_sheet_name])
                lottie3_placeholder = st.empty()

                # hanya satu periode tahun, gak usah pilih tahun
                if len(available_years) == 1:
                    selected_year = available_years
                    all_year = True
                    display_lottie_bot(lottie3_placeholder, position="right", height=197, width=300, key="blink1")
                else: 
                    st.markdown("")
                    all_year = st.checkbox("SEMUA TAHUN", False)
                    if all_year:
                        selected_year = available_years
                    else:
                        selected_year = st.multiselect(
                            "Atau Pilih Tahun (berurutan)",
                            options=available_years,
                            placeholder="Pilih tahun...",
                            key="year_select" 
                        )
                
                # validasi tahun yang dipilih harus berurutan
                if len(selected_year) > 0 and not is_sequential_years(selected_year):
                    st.error("Tahun yang dipilih harus berurutan!")
                    st.session_state.run_processing = False 
                    return
                elif len(selected_year) == 0:
                    st.info("Silakan pilih tahun untuk memulai pemrosesan data.")
                    st.session_state.run_processing = False

            st.write("")
            if selected_keys and selected_year:
                if 'monthly_avg_map' not in st.session_state or len(st.session_state.monthly_avg_map) != len(selected_keys):
                    st.session_state.monthly_avg_map = {}
                    st.session_state.merged_df = None
                    st.session_state.run_processing = True

                # checking perubahan pada pilihan tahun dan komoditas: ada perubahan, perlu ulang proses data
                current_keys_tuple = tuple(sorted(selected_keys))
                current_years_tuple = tuple(sorted(selected_year))

                params_changed = (
                    current_keys_tuple != tuple(sorted(st.session_state.last_processed_keys)) or
                    current_years_tuple != tuple(sorted(st.session_state.last_processed_years))
                )

                if params_changed:
                    st.session_state.merged_df = None   
                    st.session_state.monthly_avg_map = {} 
                    st.session_state.kmeans_preview = None
                    st.session_state.dbscan_preview = None

                if st.session_state.run_processing:
                    lottie2_placeholder.empty()
                    lottie3_placeholder.empty()
                    multi_commodities = []
                    
                    with st.expander("DATA PREVIEW"):
                        lottie_placeholder = st.empty() 
                        display_loading_lottie(lottie_placeholder, height=500, width=500, key="dataset_loader") 

                        tabs_display = selected_display.copy()
                        tabs = st.tabs(tabs_display)
                    
                        for i, sheet_key in enumerate(selected_keys):
                            current_df = df[sheet_key].copy() 
                            display_name = selected_display[i]
                            filtered_df = filter_df_by_years(current_df, selected_year)

                            if st.session_state.merged_df is None:
                                with st.spinner("Memproses data..."):
                                    cleaned_df = data_cleaning(filtered_df)
                                    preprocessed_df = data_preprocessing(cleaned_df)
                                    st.session_state.monthly_avg_map[sheet_key] = preprocessed_df.copy()

                                    cols_to_rename = preprocessed_df.columns[1:]
                                    prefix = f"{sheet_key}_"
                                    rename_map = {col: f"{prefix}{col}" for col in cols_to_rename}

                                    df_to_merge = preprocessed_df.copy()
                                    df_to_merge.rename(columns=rename_map, inplace=True)
                                    multi_commodities.append(df_to_merge)
                            
                            with tabs[i]:
                                st.write(f"### **Data Komoditas:** {display_name}")
                                display_data(filtered_df)

                        if multi_commodities:
                            merge_join = multi_commodities[0].columns[0]
                            
                            st.session_state.merged_df = reduce(
                                lambda left, right: pd.merge(
                                    left, right, 
                                    on=merge_join,
                                    how='inner',
                                    suffixes=('', '')
                                ), 
                                multi_commodities
                            )
                            
                            st.session_state.last_processed_keys = selected_keys
                            st.session_state.last_processed_years = selected_year
                            
                        lottie_placeholder.empty()

                if 'merged_df' in st.session_state and st.session_state.merged_df is not None:

                    df_input = st.session_state.merged_df

                    # checking perubahan pada data input komputasi
                    needs_recalculation = (
                        st.session_state.kmeans_preview is None or
                        st.session_state.dbscan_preview is None
                    )

                    with st.expander("CLUSTERING PREVIEW"):
                        lottie_placeholder = st.empty()
                        display_loading_lottie(lottie_placeholder, height=500, width=500, key="test_loader")

                        if needs_recalculation:
                            # with st.spinner("Clustering preview..."):
                            result_kmeans = kmeans_test(df_input.copy()) 
                            result_dbscan = dbscan_test(df_input.copy())

                            st.session_state.kmeans_preview = result_kmeans
                            st.session_state.dbscan_preview = result_dbscan
                            # st.session_state.df_input_hash = current_df_hash

                        result_kmeans = st.session_state.kmeans_preview
                        result_dbscan = st.session_state.dbscan_preview
                        display_test(result_kmeans, result_dbscan)
                        lottie_placeholder.empty()

                    default_k = int(st.session_state.kmeans_preview['n_clusters'])
                    default_eps = float(st.session_state.dbscan_preview['eps'])
                    default_minpts = int(st.session_state.dbscan_preview['min_pts'])

                    st.write("")

                    col_method, col_param = st.columns(2)
                    with col_method:
                        method = st.selectbox("Pilih Metode Clustering...", ["K-Means", "DBSCAN"])   
                    with col_param:
                        if method == "K-Means":
                            k = st.number_input("Jumlah cluster", min_value=2, max_value=10, value=default_k, step=1)
                        elif method == "DBSCAN": 
                            eps = st.number_input("Epsilon", min_value=0.1, max_value=25.0, value=default_eps, step=0.01, format="%.2f")
                            minpts = st.number_input("MinPts", min_value=2, max_value=10, value=default_minpts, step=1)

                    st.session_state.cluster_executed = None
                    if st.button("Jalankan Clustering 🚀"):
                        # with st.spinner("Sedang menjalankan clustering..."):
                        lottie_placeholder = st.empty()
                        display_loading_lottie(lottie_placeholder, height=500, width=500, key="clustering_loader")

                        if method == "K-Means":
                            kmeans_df, sil, dbi, runtime = kmeans_clustering(df_input, k)
                            st.session_state.cluster_result = (kmeans_df, sil, dbi, runtime)

                            st.session_state.current_method = method 
                            st.session_state.final_df_input = df_input.copy() 
                            st.session_state.final_k = k
                            # st.session_state.final_eps = None
                            # st.session_state.final_minpts = None

                        elif method == "DBSCAN":
                            dbscan_df, sil, dbi, runtime = dbscan_clustering(df_input, eps, minpts)
                            st.session_state.cluster_result = (dbscan_df, sil, dbi, runtime)

                            st.session_state.current_method = method 
                            st.session_state.final_df_input = df_input.copy()
                            # st.session_state.final_k = None
                            st.session_state.final_eps = eps
                            st.session_state.final_minpts = minpts

                        st.session_state.cluster_executed = True
                        st.session_state.current_method = method 
                        st.toast("✅ Clustering berhasil dijalankan!")

                    cluster_df_result = None
                    if st.session_state.get('cluster_executed') and st.session_state.cluster_result is not None:
                        cluster_df_result, sil, dbi, runtime = st.session_state.cluster_result
                        # df_input_rerun = st.session_state.final_df_input 
                        current_method = st.session_state.current_method

                        if method == "K-Means":
                            df_result = df_input.merge(cluster_df_result, on="Kabupaten/Kota", how="inner")

                        elif method == "DBSCAN":
                            df_result = df_input.merge(cluster_df_result, on="Kabupaten/Kota", how="inner")

                        current_method = st.session_state.current_method
                        lottie_placeholder.empty()

                        if cluster_df_result is not None:
                            visualize_cluster(df_result, st.session_state.monthly_avg_map, sil, dbi, runtime, current_method)

                    


                        
        except Exception as e:
            traceback.print_exc()
            if(len(selected_keys) > 0):
                st.error(f"Error: {e}")

    else:
        traceback.print_exc()
        st.error("Upload file untuk memulai proses!")


