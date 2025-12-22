"""
Streamlit web app for Iris species prediction
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained model and preprocessing objects"""
    with open('models/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, label_encoder, feature_names

def main():
    # Title
    st.markdown("# 🌸 Iris Species Predictor")
    st.markdown("Predict iris species using machine learning based on flower measurements")
    
    # Load models
    try:
        model, scaler, le, feature_names = load_models()
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run `python train_model.py` first.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Info", "📈 Example Data"])
    
    with tab1:
        st.subheader("Enter Flower Measurements")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                value=5.5,
                step=0.1
            )
            
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                value=3.5,
                step=0.1
            )
        
        with col2:
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                value=3.0,
                step=0.1
            )
            
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                value=1.0,
                step=0.1
            )
        
        # Make prediction
        if st.button("🔮 Predict Species", use_container_width=True):
            # Prepare input
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            
            # Get prediction and probabilities
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            predicted_species = le.classes_[prediction]
            confidence = probabilities[prediction] * 100
            
            # Display results
            st.success(f"### Predicted Species: **{predicted_species}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{confidence:.2f}%", delta=f"{confidence-50:.1f}%")
            
            # Display all probabilities
            st.markdown("#### Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Species': le.classes_,
                'Probability': [f"{p*100:.2f}%" for p in probabilities]
            })
            
            # Create visualization of probabilities
            fig = px.bar(
                x=le.classes_,
                y=probabilities * 100,
                labels={'x': 'Species', 'y': 'Probability (%)'},
                title='Prediction Confidence by Species',
                color=probabilities * 100,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display input summary
            st.markdown("#### Input Features Summary")
            input_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': [sepal_length, sepal_width, petal_length, petal_width]
            })
            st.table(input_df)
    
    with tab2:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", "Support Vector Machine (SVM)")
            st.metric("Features Used", len(feature_names))
            st.metric("Output Classes", len(le.classes_))
        
        with col2:
            st.info(f"**Iris Species**: {', '.join(le.classes_)}")
            st.info(f"**Features**: {', '.join(feature_names)}")
        
        st.markdown("#### About the Model")
        st.write("""
        - **Algorithm**: Support Vector Machine (SVM) with RBF kernel
        - **Training Data**: Iris dataset (150 samples)
        - **Train/Test Split**: 80/20
        - **Feature Scaling**: StandardScaler
        - **Hyperparameter Tuning**: GridSearchCV
        """)
    
    with tab3:
        st.subheader("Example Predictions")
        
        # Example data
        examples = {
            'Setosa (Typical)': [5.1, 3.5, 1.4, 0.2],
            'Versicolor (Typical)': [5.9, 2.7, 4.2, 1.3],
            'Virginica (Typical)': [6.7, 3.0, 5.5, 2.0],
        }
        
        for name, values in examples.items():
            input_data = np.array([values])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]
            pred_species = le.classes_[pred]
            confidence = probs[pred] * 100
            
            with st.expander(f"📋 {name}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Sepal Length**: {values[0]} cm")
                    st.write(f"**Sepal Width**: {values[1]} cm")
                    st.write(f"**Petal Length**: {values[2]} cm")
                    st.write(f"**Petal Width**: {values[3]} cm")
                with col2:
                    st.metric("Predicted", pred_species)
                    st.metric("Confidence", f"{confidence:.2f}%")

if __name__ == '__main__':
    main()
