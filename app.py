import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DataX - Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-top: 0rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .insight-box {
        background-color: #E0F2FE;
        border-left: 5px solid #0284C7;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
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

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 📊 DataX Dashboard")
    st.markdown("---")
    
    menu = option_menu(
        menu_title="Main Menu",
        options=["Data Overview", "EDA", "Regression Analysis", "Clustering", "Anomaly Detection"],
        icons=["house", "bar-chart", "graph-up", "diagram-3", "exclamation-triangle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#f59e0b", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#e2e8f0"},
            "nav-link-selected": {"background-color": "#0284c7"},
        }
    )
    
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)

# ==========================================
# MAIN CONTENT AREA
# ==========================================
df = st.session_state.df

st.markdown('<p class="main-header">DataX Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Data Science Workspace</p>', unsafe_allow_html=True)

if df is None:
    st.info("👈 Please upload a CSV dataset from the sidebar to get started.")
else:
    # ---------------------------------------------------------
    # MODULE 1: DATA OVERVIEW
    # ---------------------------------------------------------
    if menu == "Data Overview":
        st.header("Data Health & Overview")
        
        # 1. Health Score Calculation
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        health_score = 100 - ((missing_cells / total_cells) * 100) - ((duplicate_rows / df.shape[0]) * 100)
        health_score = max(0, min(100, health_score)) # Clamp between 0 and 100
        
        # Determine color based on score
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
            
    # ---------------------------------------------------------
    # MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ---------------------------------------------------------
    elif menu == "EDA":
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

    # ---------------------------------------------------------
    # MODULE 3: REGRESSION ANALYSIS
    # ---------------------------------------------------------
    elif menu == "Regression Analysis":
        st.header("Regression Analysis")
        st.markdown("Perform simple or multiple linear regression on numeric variables.")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Your dataset needs at least two numeric columns for regression.")
        else:
            st.markdown("### Feature Selection")
            col_x, col_y = st.columns(2)
            with col_x:
                features = st.multiselect("Select Feature(s) [X]:", numeric_cols, default=[numeric_cols[0]])
            with col_y:
                target = st.selectbox("Select Target [Y]:", [c for c in numeric_cols if c not in features])
                
            if features and target:
                # Data Preparation
                reg_df = df[features + [target]].dropna()
                if len(reg_df) < len(df):
                    st.info(f"Dropped {len(df) - len(reg_df)} rows containing missing values for regression.")
                
                X = reg_df[features]
                y = reg_df[target]
                
                # Model Training
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                # Metrics Display
                st.markdown("### Model Performance")
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("Intercept", f"{model.intercept_:.4f}")
                m3.metric("No. of Features", len(features))
                
                # Insights Generation
                insight_text = ""
                if r2 > 0.8:
                    insight_text = "Excellent fit! The model shows a strong correlation and explains a high percentage of the variance in the target."
                elif r2 > 0.5:
                    insight_text = "Good fit. The model shows moderate correlation with the target variable."
                else:
                    insight_text = "Weak correlation. The selected features might not be the best predictors, or the relationship is non-linear."
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>💡 AI Insight:</strong> The R² Score is {r2:.2f}. {insight_text}
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("### Visualizations")
                tab1, tab2, tab3 = st.tabs(["Scatter Plot (Simple Reg)", "Actual vs Predicted", "Residual Plot"])
                
                with tab1:
                    if len(features) == 1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.scatterplot(x=reg_df[features[0]], y=reg_df[target], ax=ax, alpha=0.6, label='Actual Data')
                        sns.lineplot(x=reg_df[features[0]], y=y_pred, ax=ax, color='red', label='Regression Line')
                        ax.set_title(f"Regression: {features[0]} vs {target}")
                        st.pyplot(fig)
                    else:
                        st.info("Scatter plot with regression line is only available for Simple Linear Regression (1 feature selected).")
                
                with tab2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=y, y=y_pred, ax=ax, alpha=0.6)
                    
                    min_val = min(y.min(), y_pred.min())
                    max_val = max(y.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
                    
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Actual vs Predicted")
                    ax.legend()
                    st.pyplot(fig)
                    
                with tab3:
                    residuals = y - y_pred
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
                    ax.axhline(0, color='red', linestyle='--')
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residual Plot (Errors)")
                    st.pyplot(fig)
                
                # Real-time Prediction Interactive Panel
                st.markdown("### 🎛️ Interactive Prediction Simulator")
                st.markdown("Adjust the feature values to see real-time predictions.")
                
                pred_cols = st.columns(len(features))
                input_data = []
                for i, feature in enumerate(features):
                    with pred_cols[i]:
                        min_val = float(reg_df[feature].min())
                        max_val = float(reg_df[feature].max())
                        mean_val = float(reg_df[feature].mean())
                        val = st.slider(f"{feature}", min_val, max_val, mean_val, key=f"slider_{feature}")
                        input_data.append(val)
                
                input_array = np.array(input_data).reshape(1, -1)
                prediction = model.predict(input_array)[0]
                
                st.success(f"**Predicted {target}:**  `{prediction:.4f}`")

    # ---------------------------------------------------------
    # MODULE 4: K-MEANS CLUSTERING
    # ---------------------------------------------------------
    elif menu == "Clustering":
        st.header("K-Means Clustering")
        st.markdown("Automatically group your data into distinct clusters based on similarity.")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("You need at least 2 numeric columns for 2D clustering visualization.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                cluster_features = st.multiselect("Select 2 Features for Clustering:", numeric_cols, default=numeric_cols[:2])
            with col2:
                num_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=3)
                
            if len(cluster_features) == 2:
                cluster_df = df[cluster_features].dropna()
                
                # K-Means Training
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cluster_df)
                cluster_df['Cluster'] = cluster_labels
                
                st.markdown("### Clustering Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.scatterplot(
                    data=cluster_df, 
                    x=cluster_features[0], 
                    y=cluster_features[1], 
                    hue='Cluster', 
                    palette='tab10', 
                    ax=ax,
                    alpha=0.7
                )
                
                # Plot centroids
                centroids = kmeans.cluster_centers_
                ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
                
                ax.set_title(f"K-Means Clustering (K={num_clusters})")
                ax.legend()
                st.pyplot(fig)
                
                # Insights
                st.markdown(f"""
                <div class="insight-box">
                    <strong>💡 AI Insight:</strong> The algorithm has partitioned the data into {num_clusters} distinct groups based on {cluster_features[0]} and {cluster_features[1]}. The red 'X' markers represent the mathematical center (centroid) of each cluster.
                </div>
                """, unsafe_allow_html=True)
                
            elif len(cluster_features) > 2:
                st.warning("Please select exactly 2 features for 2D visualization.")
            else:
                st.warning("Please select at least 2 features.")

    # ---------------------------------------------------------
    # MODULE 5: ANOMALY DETECTION
    # ---------------------------------------------------------
    elif menu == "Anomaly Detection":
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
