import streamlit as st
import pandas as pd
from PIL import Image
import pickle

def run():
    with open('src/model_terbaik.pkl', 'rb') as file_1:
        model = pickle.load(file_1)

    st.title('Prediksi Pengunjung Berpotensi Membeli atau Tidak')

    # img_url = 'customer.png'
    # gambar = Image.open(img_url)
    # st.image(gambar)

    with st.form('pengunjung'):
        st.markdown("### Masukkan Informasi Pengunjung Website")

        Administrative = st.number_input('Jumlah halaman administratif', min_value=0, step=1, value=2)
        Administrative_Duration = st.number_input('Durasi halaman administratif (detik)', min_value=0.0, step=1.0, value=100.0)
        Informational = st.number_input('Jumlah halaman informasi', min_value=0, step=1, value=1)
        Informational_Duration = st.number_input('Durasi halaman informasi (detik)', min_value=0.0, step=1.0, value=50.0)
        ProductRelated = st.number_input('Jumlah halaman produk', min_value=0, step=1, value=10)
        ProductRelated_Duration = st.number_input('Durasi halaman produk (detik)', min_value=0.0, step=1.0, value=300.0)
        BounceRates = st.slider('Bounce Rates (0 - 1)', min_value=0.0, max_value=1.0, value=0.02, step=0.01)
        ExitRates = st.slider('Exit Rates (0 - 1)', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        PageValues = st.number_input('Nilai PageValues', min_value=0.0, value=5.0)
        SpecialDay = st.slider('Tingkat kedekatan ke hari spesial (0 - 1)', min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        month_mapping = {
            'Januari': 'Jan', 'Februari': 'Feb', 'Maret': 'Mar', 'April': 'Apr',
            'Mei': 'May', 'Juni': 'June', 'Juli': 'Jul', 'Agustus': 'Aug',
            'September': 'Sep', 'Oktober': 'Oct', 'November': 'Nov', 'Desember': 'Dec'
        }
        month_display = st.selectbox('Bulan kunjungan', list(month_mapping.keys()))
        Month = month_mapping[month_display]

        OperatingSystems = st.selectbox('Operating System', [1, 2, 3, 4, 5, 6, 7, 8])
        Browser = st.selectbox('Browser', list(range(1, 14)))
        Region = st.selectbox('Region', list(range(1, 10)))
        TrafficType = st.selectbox('Tipe Traffic', list(range(1, 21)))
        VisitorType = st.selectbox('Tipe Pengunjung', ['New_Visitor', 'Returning_Visitor', 'Other'])
        Weekend_input = st.selectbox('Apakah kunjungan terjadi di akhir pekan?', ['Iya', 'Tidak'])
        Weekend = True if Weekend_input == 'Iya' else False

        submitted = st.form_submit_button('Submit')

    pengunjung = {
        'Administrative': Administrative,
        'Administrative_Duration': Administrative_Duration,
        'Informational': Informational,
        'Informational_Duration': Informational_Duration,
        'ProductRelated': ProductRelated,
        'ProductRelated_Duration': ProductRelated_Duration,
        'BounceRates': BounceRates,
        'ExitRates': ExitRates,
        'PageValues': PageValues,
        'SpecialDay': SpecialDay,
        'Month': Month,
        'OperatingSystems': OperatingSystems,
        'Browser': Browser,
        'Region': Region,
        'TrafficType': TrafficType,
        'VisitorType': VisitorType,
        'Weekend': Weekend
    }

    df = pd.DataFrame([pengunjung])

    if submitted:
        pred = model.predict(df)
        hasil = 'Akan Membeli' if pred[0] else 'Tidak Membeli'
        st.write(f"### Hasil Prediksi: **{hasil}**")
        if hasil == 'Akan Membeli':
            st.image("https://media.giphy.com/media/8xgqLTTgWqHWU/giphy.gif")
        else:
            st.image("https://media.giphy.com/media/6IGNW4wiyU8Mw/giphy.gif")

if __name__ == '__main__':
    run()
