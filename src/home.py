import streamlit as st
import pandas as pd

def run():
    st.title("🛍️ Project Data Science: Prediksi Pembelian Pengunjung Website")
    st.markdown("---")

    # Background
    st.markdown("### Latar Belakang")
    st.markdown("""
    Website e-commerce seringkali memiliki banyak pengunjung, namun hanya sebagian kecil yang benar-benar melakukan pembelian. 
    Dengan menganalisis perilaku pengunjung, kita dapat membangun model prediktif untuk memperkirakan siapa yang kemungkinan besar akan membeli. 
    Hal ini dapat membantu tim pemasaran dalam mengarahkan promosi secara lebih efisien dan meningkatkan konversi penjualan.
    """)

    # Problem Statement
    st.markdown("### Problem Statement")
    st.markdown("""
    Meningkatkan conversion rate pembelian dari 1% ke 2.5 % dalam waktu 3 bulan, dengan cara mengembangkan model machine learning untuk mengetahui apakah seorang pengguna cenderung akan melakukan pembelian atau tidak sehingga pengguna yang diprediksi akan membeli akan diperlakukan secara berbeda dalam strategi pemasaran
    """)

    # Objective
    st.markdown("### Tujuan")
    st.markdown("""
    Membangun model machine learning untuk mengklasifikasikan pengunjung yang berpotensi membeli, 
    dengan fokus pada metrik **recall** agar pengunjung yang benar-benar membeli dapat dikenali sebanyak mungkin.
    """)

    # Model Overview
    st.markdown("### Model Overview")
    st.markdown("""
    Beberapa algoritma telah diuji, yaitu:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Decision Tree
    - Random Forest
    - XGBoost
    
    Model terbaik adalah **XGBoost**, dengan recall **0.88** pada data test.
    """)

    # Dataset Link
    st.markdown("### Informasi Dataset")
    st.markdown("Dataset dapat diakses melalui link berikut:")
    st.markdown("[Online Shoppers Purchasing Intention Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)")

    data_info = pd.DataFrame([
        ["Administrative", "Numerik", "Jumlah halaman administratif yang dikunjungi pengunjung"],
        ["Administrative_Duration", "Numerik", "Durasi waktu (detik) pada halaman administratif"],
        ["Informational", "Numerik", "Jumlah halaman informasi yang dikunjungi pengunjung"],
        ["Informational_Duration", "Numerik", "Durasi waktu (detik) pada halaman informasi"],
        ["ProductRelated", "Numerik", "Jumlah halaman produk yang dikunjungi pengunjung"],
        ["ProductRelated_Duration", "Numerik", "Durasi waktu (detik) pada halaman produk"],
        ["BounceRates", "Numerik", "Rasio pengunjung keluar dari halaman setelah melihat satu halaman"],
        ["ExitRates", "Numerik", "Rasio keluar dari halaman terakhir sebelum meninggalkan situs"],
        ["PageValues", "Numerik", "Nilai halaman berdasarkan transaksi dan navigasi sebelumnya"],
        ["SpecialDay", "Numerik", "Indikator kedekatan waktu kunjungan dengan hari spesial (misal: Valentine)"],
        ["Month", "Kategorikal", "Bulan saat kunjungan terjadi"],
        ["OperatingSystems", "Kategorikal", "Jenis sistem operasi pengunjung"],
        ["Browser", "Kategorikal", "Jenis browser yang digunakan pengunjung"],
        ["Region", "Kategorikal", "Wilayah geografis pengunjung"],
        ["TrafficType", "Kategorikal", "Sumber traffic kunjungan"],
        ["VisitorType", "Kategorikal", "Tipe pengunjung: baru, kembali, atau lain-lain"],
        ["Weekend", "Boolean", "Apakah kunjungan terjadi saat akhir pekan"],
        ["Revenue", "Boolean (Target)", "Apakah pengunjung melakukan pembelian"]
    ], columns=["Nama", "Tipe Data", "Deskripsi"])

    st.dataframe(data_info, use_container_width=True)

    # Cara Menggunakan
    st.markdown("### Cara Menggunakan Dashboard")
    st.markdown("""
    1. Buka halaman **EDA** untuk mengeksplorasi data dan pola perilaku pengunjung.
    2. Gunakan halaman **Prediksi** untuk memasukkan data dan melihat apakah pengunjung akan melakukan pembelian.
    3. Hasil prediksi dapat digunakan sebagai dasar pengambilan keputusan pemasaran berbasis data.
    """)

if __name__ == '__main__':
    run()