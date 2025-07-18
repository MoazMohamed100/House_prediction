import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load('house_price_model.pkl')
columns = joblib.load('model_features.pkl')

st.title("House Price Prediction App")

# Input fields
def user_input():
    user_data = {}
    for col in columns:
        user_data[col] = st.number_input(f"{col}", value=0)
    return pd.DataFrame([user_data])

# Make prediction
input_df = user_input()
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Sale Price: ${prediction:,.2f}")
