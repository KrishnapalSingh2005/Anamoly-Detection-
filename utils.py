import streamlit as st
import pandas as pd

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def download_csv(df, filename="data.csv"):
    return df.to_csv(index=False).encode('utf-8')
