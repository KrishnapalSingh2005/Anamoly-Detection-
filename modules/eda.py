import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render(df):
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Understand the hidden patterns and relationships within your data.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("You need at least 2 numeric columns for advanced EDA.")
    else:
        tab1, tab2 = st.tabs(["Correlation Heatmap", "Pairplot Distributions"])
        
        with tab1:
            st.markdown("### Correlation Heatmap")
            st.markdown("Discover how features correlate with each other. Values closer to 1 or -1 indicate strong relationships.")
            
            corr_df = df[numeric_cols].dropna().corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)
            
        with tab2:
            st.markdown("### Pairplot Analysis")
            st.markdown("Visualize pairwise relationships and distributions. (Select up to 5 variables to prevent overcrowding)")
            
            selected_vars = st.multiselect("Select variables for Pairplot", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            
            if len(selected_vars) > 0:
                fig = sns.pairplot(df[selected_vars].dropna(), diag_kind="kde", corner=True)
                st.pyplot(fig)
            else:
                st.info("Select variables to view the pairplot.")
