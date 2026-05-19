import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils import download_csv

def render(df):
    st.header("Anomaly Detection")
    st.markdown("Identify abnormal data points (outliers) using robust statistical methods.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for anomaly detection.")
    else:
        st.markdown("### Configuration")
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select Feature to analyze:", numeric_cols)
        with col2:
            method = st.radio("Select Method:", ["Z-Score (Threshold=3)", "IQR (Interquartile Range)", "Compare Both"])
            
        clean_df = df.dropna(subset=[selected_col]).copy()
        
        # Anomaly Computation
        if "Z-Score" in method or "Both" in method:
            mean_val = clean_df[selected_col].mean()
            std_val = clean_df[selected_col].std()
            clean_df['Z_Score'] = (clean_df[selected_col] - mean_val) / std_val
            clean_df['Anomaly_Z'] = clean_df['Z_Score'].abs() > 3
            
        if "IQR" in method or "Both" in method:
            Q1 = clean_df[selected_col].quantile(0.25)
            Q3 = clean_df[selected_col].quantile(0.75)
            IQR_val = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR_val
            upper_bound = Q3 + 1.5 * IQR_val
            clean_df['Anomaly_IQR'] = (clean_df[selected_col] < lower_bound) | (clean_df[selected_col] > upper_bound)
            
        # Method consolidation
        if "Z-Score" in method and "Both" not in method:
            anomalies = clean_df[clean_df['Anomaly_Z']]
            anomaly_col = 'Anomaly_Z'
            method_name = "Z-Score"
        elif "IQR" in method and "Both" not in method:
            anomalies = clean_df[clean_df['Anomaly_IQR']]
            anomaly_col = 'Anomaly_IQR'
            method_name = "IQR"
        else:
            anomalies = clean_df[clean_df['Anomaly_Z'] | clean_df['Anomaly_IQR']]
            anomaly_col = 'Anomaly_Both'
            clean_df['Anomaly_Both'] = clean_df['Anomaly_Z'] | clean_df['Anomaly_IQR']
            method_name = "Z-Score & IQR"

        num_anomalies = len(anomalies)
        total_rows = len(clean_df)
        perc_anomalies = (num_anomalies / total_rows) * 100 if total_rows > 0 else 0
        
        st.markdown("### Detection Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed Rows", total_rows)
        m2.metric("Anomalies Detected", num_anomalies)
        m3.metric("% of Anomalies", f"{perc_anomalies:.2f}%")
        
        # Insights
        insight_text = f"Detected **{num_anomalies} anomalies** ({perc_anomalies:.2f}% of the dataset) using the {method_name} method."
        if perc_anomalies > 5:
            insight_text += " This is a relatively high number of outliers. You may want to investigate the data collection process."
        elif perc_anomalies > 0:
            insight_text += " The number of outliers falls within a typical expected range."
        else:
            insight_text += " No outliers detected. Your data is well-distributed within the boundaries!"
            
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 AI Insight:</strong> {insight_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### Visualizations")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.write("**Boxplot (Distribution & Outliers)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=clean_df[selected_col], ax=ax, color='lightblue', flierprops=dict(markerfacecolor='red', marker='o'))
            st.pyplot(fig)
            
        with v_col2:
            st.write("**Histogram Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(clean_df[selected_col], kde=True, ax=ax, color='steelblue')
            
            if "IQR" in method or "Both" in method:
                ax.axvline(lower_bound, color='red', linestyle='--', label='IQR Bounds')
                ax.axvline(upper_bound, color='red', linestyle='--')
            if "Z-Score" in method or "Both" in method:
                ax.axvline(mean_val - 3*std_val, color='orange', linestyle=':', label='Z-Score Bounds')
                ax.axvline(mean_val + 3*std_val, color='orange', linestyle=':')
                
            ax.legend(fontsize='small')
            st.pyplot(fig)
            
        st.write("**Scatter Plot Tracking Anomalies**")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        colors = clean_df[anomaly_col].map({True: 'red', False: 'blue'})
        sizes = clean_df[anomaly_col].map({True: 40, False: 15})
        
        ax.scatter(clean_df.index, clean_df[selected_col], c=colors, s=sizes, alpha=0.6)
        ax.set_xlabel("Data Index")
        ax.set_ylabel(selected_col)
        ax.set_title("Anomalies Highlighted in RED")
        st.pyplot(fig)
        
        # Data Table Download
        st.markdown("### Anomaly Data Records")
        if num_anomalies > 0:
            st.dataframe(anomalies, use_container_width=True)
            csv = download_csv(anomalies)
            st.download_button(
                label="📥 Download Anomalies as CSV",
                data=csv,
                file_name="detected_anomalies.csv",
                mime="text/csv",
            )
        else:
            st.success("No anomalies detected in the selected feature!")
