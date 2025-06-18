# Import library
import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency
import numpy as np

# Fungsi bantu Cramer's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

def run():
    st.title('Exploratory Data Analysis')

    st.markdown('Eksplorasi data analisis (EDA) dilakukan secara bertahap dan menyeluruh untuk menemukan pola yang relevan terhadap perilaku pembelian pengguna website. Analisis ini dibagi menjadi dua bagian utama:')
    st.markdown('1. Bagian pertama berfokus pada pemahaman fitur yang paling berkorelasi dengan target, terutama peran penting dari PageValues')
    st.markdown('2. Bagian kedua menggali pengaruh faktor waktu dan karakteristik pengunjung terhadap tingkat pembelian')
    st.markdown('Pembagian ini membantu membangun alur analisis yang terstruktur dan memperjelas keterkaitan antar temuan')

    df = pd.read_csv('src/ecommerce_purchasing_intention.csv')

    fields = ['Eksplorasi data analisis bagian 1', 
              'Eksplorasi data analisis bagian 2']

    pilihan = st.selectbox('Pilih Bagian EDA', fields)

    if pilihan == fields[0]:
        st.subheader('- Bagaimana Tingkat Signifikansi antara fitur dengan target ?')
        st.markdown("Kita akan mengecek signifikasi data dengan target, yaitu kolom 'Revenue' yang bertipe data binary. Sehingga untuk melihat signifikasi antara fitur numerical ke target akan dilakukan dengan pointbiserialr sedangkan antara fitur categorical ke target akan dilakukan dengan Chi-Squared")
        num_features = [
            "Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]

        hasil = []
        for col in num_features:
            corr, p = pointbiserialr(df[col], df['Revenue'])
            hasil.append({'Fitur': col, 'P-value': p, 'Tingkat signifikasi': corr})

        point_biserial = pd.DataFrame(hasil)
        point_biserial_sorted = point_biserial.sort_values(by='Tingkat signifikasi', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=point_biserial_sorted, x='Tingkat signifikasi', y='Fitur', palette='coolwarm', ax=ax)
        ax.set_title('Point-Biserial Correlation antara Fitur Numerik dan Revenue')
        ax.set_xlabel('Korelasi')
        ax.grid(axis='x')
        st.pyplot(fig)
        st.markdown('Hasilnya H1 diterima, dimana semua fitur numerical memiliki p-value < 0.05, namun fitur-fitur ini tidak ada yang memiliki tingkat signifikasi lebih dari 0.5, sehingga nantinya perlu dilakukan feature engineering untuk menghasilkan fitur-fitur baru yang lebih informatif')

        cat_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
        hasil = []

        for col in cat_features:
            contingency = pd.crosstab(df[col], df['Revenue'])
            chi2, p, dof, _ = chi2_contingency(contingency)
            cramers = cramers_v(contingency)
            hasil.append({'Fitur': col, 'P-value': p, "Cramer's V": cramers})

        chi_square_df = pd.DataFrame(hasil)
        sorted_cramer = chi_square_df.sort_values(by="Cramer's V", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sorted_cramer, x="Cramer's V", y='Fitur', color='royalblue', ax=ax2)
        ax2.set_title("Cramér's V antara Fitur Kategorikal dan Revenue")
        ax2.set_xlabel("Cramér's V")
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig2)
        st.markdown("Hasilnya terdapat 1 kolom yang H1 nya ditolak, yaitu kolom Region yang memiliki p-value 0.321425. Sedangkan kolom yang lain memiliki P-value < 0.05 sehingga H1 diterima. Tingkat signifikasi yang diuji menggunakan Cramér's V juga sangat rendah, sehingga perlu adanya fitur baru yang memiliki signifikasi yang besar")

        st.subheader("- Bagaimana persebaran data PageValues antara target positif dan negatif ?")
        st.markdown('Seperti yang telah diketahui pada analisis sebelumnya, fitur PageValues menunjukkan signifikasi tertinggi terhadap target Revenue di antara seluruh fitur terutama fitur numerik yang dianalisis.')
        st.markdown("Oleh karena itu, pada tahap ini kita ingin menyelidiki lebih lanjut bagaimana distribusi nilai PageValues pada dua kelompok target yaitu antara pengguna yang melakukan pembelian (Revenue = True) dan yang tidak (Revenue = False)")
        
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x='PageValues', hue='Revenue', bins=50, element='step', stat='density', common_norm=False, ax=ax3)
        ax3.set_title('Distribusi PageValues terhadap Revenue')
        ax3.set_xlim(0, df['PageValues'].quantile(0.99))
        ax3.grid(True)
        st.pyplot(fig3)

        st.markdown("Berdasarkan hasil visualisasi distribusi PageValues, terlihat bahwa sebagian besar sesi (baik yang menghasilkan pembelian atau tidak) memiliki nilai PageValues yang sangat rendah. Artinya, mayoritas pengguna hanya mengunjungi halaman-halaman yang tidak terlalu berkaitan dengan transaksi. Namun, ketika kita lihat sesi dengan Revenue = True (pengguna yang membeli), distribusinya lebih menyebar ke nilai PageValues yang lebih tinggi. Ini menunjukkan bahwa pengguna yang akhirnya melakukan pembelian cenderung menjelajahi halaman-halaman yang lebih penting secara bisnis")
        st.markdown("Sehingga, semakin tinggi nilai PageValues dalam sebuah sesi, semakin besar kemungkinan sesi tersebut berujung pada pembelian")
        
        st.subheader("- Apa Jenis Halaman yang Paling Berkontribusi terhadap Nilai PageValues ?")
        st.markdown("Setelah kita mengetahui bahwa PageValues memiliki korelasi tertinggi terhadap Revenue, dan melihat perbedaan distribusinya antara pengguna yang membeli dan tidak, pertanyaan selanjutnya adalah:")
        st.markdown("Bagian mana dari page (administratif, informasional, atau produk) yang paling berperan dalam menghasilkan nilai halaman tinggi (PageValues)?")

        eda3 = df.copy()
        eda3['Admin_Engagement'] = eda3['Administrative'] + eda3['Administrative_Duration']
        eda3['Info_Engagement'] = eda3['Informational'] + eda3['Informational_Duration']
        eda3['Product_Engagement'] = eda3['ProductRelated'] + eda3['ProductRelated_Duration']

        def dominant_section(row):
            sections = {
                'Administrative': row['Admin_Engagement'],
                'Informational': row['Info_Engagement'],
                'ProductRelated': row['Product_Engagement']
            }
            return max(sections, key=sections.get)

        eda3['DominantSection'] = eda3.apply(dominant_section, axis=1)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=eda3[eda3['PageValues'] > 0], x='DominantSection', y='PageValues', ax=ax4)
        ax4.set_title('Distribusi PageValues Berdasarkan Jenis Halaman Dominan')
        ax4.grid(True, axis='y')
        st.pyplot(fig4)

        st.markdown("Hasil visualisasi menunjukkan bahwa sesi dengan halaman produk (ProductRelated) memiliki persebaran PageValues yang paling luas, serta mengandung banyak nilai outlier yang tinggi. Artinya, pengguna yang paling banyak mengakses halaman produk cenderung berpotensi lebih besar melakukan pembelian")
        st.markdown("Sementara itu, sesi yang didominasi halaman administratif (Administrative) juga menunjukkan persebaran yang cukup tinggi, tetapi tidak sebanyak halaman produk. Di sisi lain, sesi yang paling banyak berinteraksi dengan halaman informasi (Informational) memiliki distribusi PageValues yang relatif rendah dan lebih terkonsentrasi di nilai-nilai kecil")
        st.markdown("Sehingga halaman produk adalah halaman yang paling krusial dalam mendorong pengguna menuju pembelian, dibandingkan dua jenis halaman lainnya")

        

    elif pilihan == fields[1]:
        st.subheader("- Kapan Konversi Pembelian Paling Banyak Terjadi Berdasarkan Bulan?")
        st.markdown("Salah satu faktor penting dalam perilaku pengguna e-commerce adalah waktu kunjungan, khususnya bulan. Pola musiman seperti bulan promosi, akhir tahun, acara keagamaan, dll sering kali memengaruhi perilaku pembelian pengguna. Oleh karena itu, memahami bulan-bulan apa saja yang memiliki tingkat konversi tinggi dapat membantu dalam merancang strategi pemasaran")

        month_revenue = (
            df.groupby('Month')['Revenue'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
        )
        month_revenue_true = month_revenue[month_revenue['Revenue'] == True]
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig5, ax5 = plt.subplots(figsize=(8,5))
        sns.barplot(data=month_revenue_true, x='Month', y='Percentage', order=month_order, color='skyblue', ax=ax5)
        ax5.set_title('Convertion Rate Berdasarkan Bulan')
        ax5.set_ylabel('Convertion Rate (%)')
        ax5.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig5)

        st.markdown("Dari grafik yang ditampilkan, terlihat bahwa persentase pembelian meningkat secara konsisten dari bulan mei hingga mencapai puncaknya di bulan November, dengan angka di lebih dari 25%. Sebaliknya, bulan Januari hingga April mencatatkan persentase pembelian yang sangat rendah")
        st.markdown("Pola ini mengindikasikan adanya perilaku musiman di mana pengguna lebih aktif melakukan pembelian menjelang akhir tahun. kemungkinan karena promosi seperti natal dan liburan akhir tahun")

        st.subheader("- Apakah Pembelian Lebih Sering Terjadi di Akhir Pekan pada Bulan-bulan Tertentu?")
        st.markdown("Setelah kita melihat bahwa pembelian meningkat tajam pada bulan-bulan akhir tahun seperti Oktober dan November, pertanyaan lanjutan yang muncul adalah:")
        st.markdown("Apakah pembelian tersebut lebih banyak terjadi saat akhir pekan (Weekend = True) dibanding hari biasa?")
        st.markdown("Oleh karena itu, kita akan melihat top 5 convertion rate berdasarkan bulan dan status weekend dengan persentase pembelian tertinggi")
        
        pivot = df.groupby(['Month', 'Weekend'])['Revenue'].mean().mul(100).reset_index().pivot(index='Month', columns='Weekend', values='Revenue')
        pivot = pivot.reindex(['Jul', 'Aug', 'Sep', 'Oct', 'Nov'])

        fig6, ax6 = plt.subplots(figsize=(7,5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, ax=ax6)
        ax6.set_title("Conversion Rate Berdasarkan Bulan dan Status Weekend")
        st.pyplot(fig6)

        st.markdown("Berdasarkan hasil visualisasi heatmap, pada bulan juli paling sering terjadi transaksi pada weekend, bulan agustus paling sering terjadi transaksi pada weekday, bulan september paling sering terjadi transaksi pada weekend, bulan oktober paling sering terjadi transaksi pada weekday, bulan november paling sering terjadi transaksi pada weekend, dan bulan juli paling sering terjadi transaksi pada weekend")
        st.markdown("Sehingga tidak selalu di akhir pekan pembelian sering terjadi")

        st.subheader("- Bagaimana persentase pembelian berdasarkan tipe pengunjung ?")
        st.markdown("Setelah menganalisis waktu kunjungan, kita kini ingin mengetahui bagaimana persentase pembelian berdasarkan tipe pengunjung (VisitorType). Fitur VisitorType mengelompokkan pengunjung menjadi 3 kategori yaitu, New_Visitor, Returning_Visitor, dan Other")
       
        visitor_types = ['New_Visitor', 'Other', 'Returning_Visitor']
        fig7, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, vtype in enumerate(visitor_types):
            subset = df[df['VisitorType'] == vtype]
            counts = subset['Revenue'].value_counts(normalize=True) * 100
            sizes = [counts.get(False, 0), counts.get(True, 0)]
            labels = ['Tidak Beli', 'Beli']
            axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', explode=[0, 0.08], startangle=90, textprops={'fontsize': 15})
            axes[i].set_title(f'{vtype}', fontsize=15)

        fig7.suptitle('Persentase Pembelian per Tipe Pengunjung', fontsize=20, fontweight='bold')
        st.pyplot(fig7)

        st.markdown("Berdasarkan grafik, terlihat bahwa pengunjung baru justru memiliki tingkat persentase pembelian tertinggi, yaitu sekitar 25%. Ini merupakan penemuan unik, karena secara umum dalam e-commerce, pengunjung lama dianggap lebih bernilai")
        st.markdown("Sementara itu, tipe Other menempati posisi tengah dengan tingkat persentase pembelian sekitar 19%, dan Returning_Visitor justru memiliki tingkat persentase pembelian terendah, yaitu hanya sekitar 14%. Hal ini bisa mengindikasikan bahwa pengguna yang kembali tidak langsung melakukan pembelian")

        st.subheader("- Bagaimana Pola Pembelian Pengunjung Baru Sepanjang Tahun ?")
        st.markdown("Setelah sebelumnya ditemukan bahwa pengunjung baru memiliki tingkat konversi tertinggi dibandingkan tipe pengunjung lainnya, muncul pertanyaan lanjutan")
        st.markdown("Kapan bulan terbaik untuk mendorong pembelian dari pengunjung baru lebih banyak lagi?")
        st.markdown("Sehingga analisis ini difokuskan pada pola pembelian pengunjung baru berdasarkan bulan. Dengan memahami kapan mereka paling aktif melakukan transaksi, kita dapat merancang strategi promosi yang terarah ")
        new_visitor = df[df['VisitorType'] == 'New_Visitor']
        new_visitor_month = (
            new_visitor.groupby('Month')['Revenue'].mean().mul(100).reset_index().rename(columns={'Revenue': 'Persentase_Pembelian'})
        )
        new_visitor_month['Month'] = pd.Categorical(new_visitor_month['Month'], categories=month_order, ordered=True)
        new_visitor_month = new_visitor_month.sort_values('Month')

        fig8, ax8 = plt.subplots(figsize=(8,5))
        sns.barplot(data=new_visitor_month, x='Month', y='Persentase_Pembelian', color='skyblue', ax=ax8)
        ax8.set_title('Convertion Rate Pembelian New Visitor Berdasarkan Bulan')
        ax8.set_ylabel('Covertion Rate (%)')
        ax8.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig8)
        st.markdown("Berdasarkan grafik, terlihat adanya pola peningkatan bertahap dalam convertion rate pelanggan baru sepanjang tahun. Pembelian mulai meningkat secara signifikan sejak bulan maret, namun tidak ada transaksi pada bulan april, lalu menanjak lagi dari bulan mei hingga mencapai puncaknya pada bulan December. Setelah itu, terjadi penurunan kembali pada bulan Januari")

if __name__ == '__main__':
    run()
