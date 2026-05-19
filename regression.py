import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def render(df):
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
