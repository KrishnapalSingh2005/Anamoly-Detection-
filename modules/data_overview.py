import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render(df):
    st.header("Data Health & Overview")
    
    # 1. Health Score Calculation
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    health_score = 100 - ((missing_cells / total_cells) * 100) - ((duplicate_rows / df.shape[0]) * 100)
    health_score = max(0, min(100, health_score))
    
    score_color = "green" if health_score >= 90 else "orange" if health_score >= 70 else "red"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #f8fafc; border-radius: 10px; margin-bottom: 20px; border: 2px solid {score_color};">
        <h2 style="margin:0; color: #1e293b;">Data Quality Health Score</h2>
        <h1 style="margin:0; color: {score_color}; font-size: 3rem;">{health_score:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

    # 2. Advanced Metrics Dashboard
    st.subheader("Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{df.shape[0]:,}")
    m2.metric("Total Columns", f"{df.shape[1]:,}")
    m3.metric("Missing Values", f"{missing_cells:,}")
    m4.metric("Duplicate Rows", f"{duplicate_rows:,}")
    
    st.markdown("---")
    
    # 3. Missing Values Visualization
    st.subheader("Missing Values Distribution")
    missing_srs = df.isnull().sum()
    missing_srs = missing_srs[missing_srs > 0].sort_values(ascending=True)
    
    if not missing_srs.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, len(missing_srs)*0.4)))
        missing_srs.plot(kind='barh', color='salmon', ax=ax)
        ax.set_title("Number of Missing Values per Column")
        ax.set_xlabel("Count of Missing Values")
        st.pyplot(fig)
    else:
        st.success("🎉 Perfect! There are absolutely no missing values in your dataset.")
        
    st.markdown("---")
        
    # 4. Statistical Summary
    st.subheader("Statistical Summary (Numerical Columns)")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().T, use_container_width=True)
    else:
        st.info("No numerical columns found to generate statistics.")
        
    st.markdown("---")
    
    # 5. Raw Data Expander
    with st.expander("🔍 View Raw Dataset Preview"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Showing top 100 rows.")
