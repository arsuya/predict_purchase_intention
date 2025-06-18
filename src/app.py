import streamlit as st
import eda, predict, home

def add_custom_features(df):
    df = df.copy()
    df['total_pages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['total_duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['engagement_score_pages'] = df['PageValues'] + df['total_pages']
    df['engagement_score_duration'] = df['PageValues'] + df['total_duration']
    return df

st.set_page_config(page_title='VISUALISASI MILESTOME 2',
                   layout='centered',
                   initial_sidebar_state='expanded')

with st.sidebar:
    st.write('# Navigation')
    navigation = st.radio('Page', ['Home', 'EDA', 'Prediksi'])
    st.markdown("---")

    # Kontak
    st.markdown("Project Data Science oleh<br><a href='https://www.linkedin.com/in/arvinwibowo/'>Arvin Surya Wibowo</a>", unsafe_allow_html=True)


if navigation == 'Home':
    home.run()
elif navigation == 'EDA':
    eda.run()
else:
    predict.run()