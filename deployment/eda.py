import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_eda = pd.read_csv('healthcare_noshows_appointments.csv')  # Ganti 'data.csv' dengan nama file CSV Anda

# Membuat fungsi untuk mengelompokkan usia dengan nama yang lebih deskriptif
def categorize_age(age):
    if age <= 12:
        return 'Anak-anak (0-12)'
    elif 13 <= age <= 19:
        return 'Remaja (13-19)'
    elif 20 <= age <= 35:
        return 'Dewasa Muda (20-35)'
    elif 36 <= age <= 55:
        return 'Dewasa (36-55)'
    else:
        return 'Lanjut Usia (56+)'

# Fungsi untuk menampilkan EDA
def run():
    st.title('Exploratory Data Analysis (EDA) - Patient Show Up Predictor')

    # Kategori Usia
    st.header('Distribusi Kategori Usia Pasien Berdasarkan Kehadiran')
    df_eda['Age_Group'] = df_eda['Age'].apply(categorize_age)
    age_group_counts = df_eda.groupby(['Age_Group', 'Showed_up']).size().unstack(fill_value=0)
    age_group_counts['Jumlah_Tidak_Hadir'] = age_group_counts[False]
    sorted_age_groups = age_group_counts.sort_values(by='Jumlah_Tidak_Hadir', ascending=False).index

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_eda, x='Age_Group', hue='Showed_up', order=sorted_age_groups, palette='Set2')
    plt.title('Distribusi Kategori Usia Pasien Berdasarkan Kehadiran pada Janji Medis', fontsize=14)
    plt.xlabel('Kelompok Usia', fontsize=12)
    plt.ylabel('Jumlah Pasien', fontsize=12)
    plt.legend(title='Kehadiran', labels=['Tidak Hadir', 'Hadir'])
    st.pyplot(plt)

    # Persentase Jenis Kelamin
    st.header('Persentase Jenis Kelamin Pasien Berdasarkan Kehadiran')
    gender_counts = df_eda.groupby(['Gender', 'Showed_up']).size().unstack(fill_value=0)
    gender_counts['Jumlah_Tidak_Hadir'] = gender_counts[False]
    sorted_gender = gender_counts.sort_values(by='Jumlah_Tidak_Hadir', ascending=False)
    sorted_gender.index = sorted_gender.index.map({'F': 'Female', 'M': 'Male'})

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(data=sorted_gender.reset_index().melt(id_vars='Gender', value_vars=[True, False]),
                           x='Gender', y='value', hue='Showed_up', palette=['#ff69b4', '#1f77b4'])
    plt.title('Persentase Jenis Kelamin Pasien Berdasarkan Kehadiran pada Janji Medis', fontsize=14)
    plt.xlabel('Jenis Kelamin', fontsize=12)
    plt.ylabel('Jumlah Pasien', fontsize=12)
    plt.legend(title='Kehadiran', labels=['Hadir', 'Tidak Hadir'], loc='upper right')
    st.pyplot(plt)

    # Rasio Kehadiran Berdasarkan Status Beasiswa
    st.header('Rasio Kehadiran Pasien Berdasarkan Status Beasiswa')
    scholarship_counts = df_eda.groupby(['Scholarship', 'Showed_up']).size().unstack(fill_value=0)
    scholarship_counts['Rasio_Hadir'] = scholarship_counts[True] / (scholarship_counts[True] + scholarship_counts[False]) * 100
    scholarship_counts.reset_index(inplace=True)
    scholarship_counts['Scholarship'] = scholarship_counts['Scholarship'].map({True: 'Dengan Beasiswa', False: 'Tanpa Beasiswa'})

    plt.figure(figsize=(8, 5))
    bar_plot = sns.barplot(x='Scholarship', y='Rasio_Hadir', hue='Scholarship', data=scholarship_counts, palette=['#1f77b4', '#ff69b4'], legend=False)
    plt.title('Rasio Kehadiran Pasien Berdasarkan Status Beasiswa', fontsize=14)
    plt.xlabel('Status Beasiswa', fontsize=12)
    plt.ylabel('Rasio Kehadiran (%)', fontsize=12)

    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=10)
    st.pyplot(plt)

    # Distribusi Selisih Waktu
    st.header('Distribusi Selisih Waktu berdasarkan Status Kehadiran')
    bins = [-1, 0, 3, 7, 14, float('inf')]  # Batas-batas untuk bin
    labels = ['0 hari', '1-3 hari', '4-7 hari', '8-14 hari', '15 hari atau lebih']
    df_eda['Date.diff_group'] = pd.cut(df_eda['Date.diff'], bins=bins, labels=labels)

    plt.figure(figsize=(10, 6))
    hist_plot = sns.histplot(data=df_eda, x='Date.diff_group', hue='Showed_up', 
                             multiple="dodge", palette={True: '#1f77b4', False: '#ff69b4'}, 
                             alpha=0.7)
    plt.title('Distribusi Selisih Waktu berdasarkan Status Kehadiran', fontsize=16)
    plt.xlabel('Kelompok Selisih Waktu (Hari)', fontsize=14)
    plt.ylabel('Frekuensi', fontsize=14)
    plt.legend(title='Status Kehadiran', labels=['Hadir', 'Tidak Hadir'])
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Proporsi Kehadiran Berdasarkan SMS
    st.header('Proporsi Kehadiran Berdasarkan Pengingat SMS yang Diterima')
    sms_counts = df_eda.groupby('SMS_received')['Showed_up'].value_counts().unstack(fill_value=0)
    sms_counts['Total'] = sms_counts.sum(axis=1)
    sms_counts['Hadir_Proporsi'] = sms_counts[True] / sms_counts['Total'] * 100
    sms_counts['Tidak_Hadir_Proporsi'] = sms_counts[False] / sms_counts['Total'] * 100

    plot_data = sms_counts[['Hadir_Proporsi', 'Tidak_Hadir_Proporsi']].reset_index()
    plot_data = plot_data.melt(id_vars='SMS_received', value_vars=['Hadir_Proporsi', 'Tidak_Hadir_Proporsi'],
                                var_name='Kehadiran', value_name='Proporsi')

    plt.figure(figsize=(8, 5))
    bar_plot = sns.barplot(x='SMS_received', y='Proporsi', hue='Kehadiran', data=plot_data, palette=['#1f77b4', '#d62728'])
    plt.title('Proporsi Kehadiran Berdasarkan Pengingat SMS yang Diterima', fontsize=14)
    plt.xlabel('SMS Diterima', fontsize=12)
    plt.ylabel('Proporsi (%)', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Tidak', 'Ya'], rotation=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color='#1f77b4'), plt.Rectangle((0, 0), 1, 1, color='#d62728')]
    plt.legend(handles, ['Hadir', 'Tidak Hadir'], title='Kehadiran', loc='upper right')
    plt.grid(axis='y', linestyle='--')
    st.pyplot(plt)

if __name__ == '__main__':
    run()
