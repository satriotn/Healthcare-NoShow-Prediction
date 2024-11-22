import streamlit as st
import eda  # Pastikan ini adalah nama file EDA tanpa ekstensi .py
import predict  # Pastikan ini adalah nama file prediction tanpa ekstensi .py

# Sidebar untuk navigasi
navigation = st.sidebar.selectbox('Choose Page', ('Prediction', 'EDA'))

if navigation == 'Prediction':
    predict.run()
elif navigation == 'EDA':
    eda.run()

#mantap