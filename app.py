import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from modules.utils import load_data
from modules import data_overview, eda, regression, clustering, anomaly

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
    # Route to the appropriate module based on the menu selection
    if menu == "Data Overview":
        data_overview.render(df)
    elif menu == "EDA":
        eda.render(df)
    elif menu == "Regression Analysis":
        regression.render(df)
    elif menu == "Clustering":
        clustering.render(df)
    elif menu == "Anomaly Detection":
        anomaly.render(df)
