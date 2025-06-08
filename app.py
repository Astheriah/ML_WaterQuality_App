import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model
modelo = joblib.load('best_random_forest.pkl')

# Custom styles
st.markdown("""
    <style>
    .main {
        background-color: #f0f6fc;
    }
    .stButton>button {
        background-color: #0072c6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stNumberInput>div>input {
        border-radius: 8px;
        border: 1px solid #0072c6;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tab interface ---
tabs = st.tabs(["Prediction", "Extra information"])

with tabs[0]:
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.title('💧 Water Potability Prediction')
    st.write('''
    This app uses a Machine Learning model to predict if water is **potable** or **not potable**.
    \nPlease enter the water characteristics:
    ''')
    with st.form("form_prediction"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0, help="Acidity/basicity level of water")
            Hardness = st.number_input('Hardness', min_value=0.0, value=150.0, help="Water hardness")
            Solids = st.number_input('Solids', min_value=0.0, value=10000.0, help="Total dissolved solids")
        with col2:
            Chloramines = st.number_input('Chloramines', min_value=0.0, value=7.0, help="Chloramines level")
            Sulfate = st.number_input('Sulfate', min_value=0.0, value=333.0, help="Sulfate level")
            Conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0, help="Electrical conductivity")
        with col3:
            Organic_carbon = st.number_input('Organic carbon', min_value=0.0, value=10.0, help="Organic carbon")
            Trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=60.0, help="Trihalomethanes level")
            Turbidity = st.number_input('Turbidity', min_value=0.0, value=3.0, help="Water turbidity")
        submitted = st.form_submit_button('Predict')
    if 'submitted' in locals() and submitted:
        data = pd.DataFrame({
            'ph': [ph],
            'Hardness': [Hardness],
            'Solids': [Solids],
            'Chloramines': [Chloramines],
            'Sulfate': [Sulfate],
            'Conductivity': [Conductivity],
            'Organic_carbon': [Organic_carbon],
            'Trihalomethanes': [Trihalomethanes],
            'Turbidity': [Turbidity]
        })
        prediction = modelo.predict(data)[0]
        st.markdown("---")
        if prediction == 1:
            st.success('💧 The water IS potable. Safe for consumption!')
        else:
            st.error('⚠️ The water is NOT potable. Not recommended for consumption.')
        st.markdown("---")
        st.write("**Input values:**")
        st.dataframe(data.T, use_container_width=True)

with tabs[1]:
    # Feature importance
    st.subheader("Feature Importance")
    importances = modelo.feature_importances_
    columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    fig1, ax1 = plt.subplots()
    sns.barplot(x=importances, y=columns, ax=ax1, color=sns.color_palette("Blues_r", n_colors=1)[0])
    ax1.set_title('Feature Importance')
    st.pyplot(fig1, use_container_width=True)
