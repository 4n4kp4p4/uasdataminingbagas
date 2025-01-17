import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Memuat dataset
path_data = 'Clustering.csv'  # Ganti dengan path dataset Anda
dataframe = pd.read_csv(path_data)

# Aplikasi Streamlit
st.title("Analisis Demografi & K-Means Clustering")

# Menu Sidebar
menu = st.sidebar.selectbox("Menu", ["Gambaran Umum", "Visualisasi", "K-Means Clustering"])

if menu == "Gambaran Umum":
    st.subheader("Gambaran Umum Dataset")
    st.write("### 5 Baris Pertama Dataset")
    st.write(dataframe.head())

    st.write("### Informasi Dataset")
    df_info = pd.DataFrame({
        "Column": dataframe.columns,
        "Non-Null Count": dataframe.notnull().sum(),
        "Dtype": dataframe.dtypes
    }).reset_index(drop=True)
    st.write(df_info)

    st.write("### Nilai yang Hilang")
    st.write(dataframe.isnull().sum())

if menu == "Visualisasi":
    st.subheader("Visualisasi")

    # Distribusi usia
    if st.checkbox("Tampilkan Distribusi Usia"):
        st.write("### Distribusi Usia")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(dataframe['Age'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Distribusi Usia")
        st.pyplot(fig)

    # Distribusi pendapatan
    if st.checkbox("Tampilkan Distribusi Pendapatan"):
        st.write("### Distribusi Pendapatan")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(dataframe['Income'], bins=20, kde=True, color='lightgreen', ax=ax)
        ax.set_title("Distribusi Pendapatan")
        st.pyplot(fig)

    # Boxplot berdasarkan status perkawinan
    if st.checkbox("Tampilkan Boxplot Pendapatan berdasarkan Status Perkawinan"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Marital status', y='Income', data=dataframe, ax=ax, palette='Set3')
        ax.set_title("Boxplot Pendapatan berdasarkan Status Perkawinan")
        st.pyplot(fig)

if menu == "K-Means Clustering":
    st.subheader("K-Means Clustering")

    # Memilih fitur untuk clustering
    st.write("### Pilih Fitur untuk Clustering")
    fitur_numerik = [kolom for kolom in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[kolom])]

    # Validasi nilai default
    default_values = ['Age', 'Income']
    valid_defaults = [value for value in default_values if value in fitur_numerik]

    fitur = st.multiselect("Fitur", options=fitur_numerik, default=valid_defaults)

    if len(fitur) > 1:
        # Menyiapkan data untuk clustering
        data = dataframe.dropna(subset=fitur)
        X = data[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Memilih jumlah cluster
        st.write("### Pilih Jumlah Cluster")
        n_cluster = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)

        # Menerapkan K-Means
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X_scaled)

        # Menampilkan hasil clustering
        st.write("### Pusat Cluster")
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=fitur))

        st.write("### Data dengan Label Cluster")
        st.write(data[['Cluster'] + fitur].head())

        # Visualisasi cluster untuk kombinasi dua fitur
        st.write("### Visualisasi untuk Kombinasi Fitur")
        if len(fitur) >= 2:
            feature_pairs = [(fitur[i], fitur[j]) for i in range(len(fitur)) for j in range(i + 1, len(fitur))]
            for pair in feature_pairs:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=pair[0], y=pair[1], hue='Cluster', data=data, palette='tab10', ax=ax)
                ax.set_title(f"K-Means Clustering: {pair[0]} dan {pair[1]}")
                st.pyplot(fig)
    else:
        st.warning("Pilih setidaknya 2 fitur numerik untuk clustering.")
