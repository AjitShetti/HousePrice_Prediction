import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_record, load_data

@st.cache_data
def load_model_and_columns():
    """Load trained model, scaler and feature column list."""
    model = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    X_train, _, _, _, _ = load_data()
    return model, scaler, X_train.columns.tolist()


model, scaler, feature_columns = load_model_and_columns()

# read raw dataset for dropdowns
raw_df = pd.read_csv("data/Bengaluru_House_Data.csv")

area_types = raw_df['area_type'].dropna().unique().tolist()
availabilities = raw_df['availability'].dropna().unique().tolist()
locations = raw_df['location'].dropna().unique().tolist()
sizes = raw_df['size'].dropna().unique().tolist()

st.title("Bengaluru House Price Predictor")

with st.form("input_form"):
    area_type = st.selectbox("Area type", area_types)
    availability = st.selectbox("Availability", availabilities)
    location = st.selectbox("Location", locations)
    size = st.selectbox("Size", sizes)
    bath = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)
    balcony = st.number_input("Balconies", min_value=0, max_value=10, value=1)
    total_sqft = st.number_input("Total sqft", min_value=0.0, value=500.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "area_type": area_type,
        "availability": availability,
        "location": location,
        "size": size,
        "bath": bath,
        "balcony": balcony,
        "total_sqft": total_sqft,
        "price": 0
    }
    df = preprocess_record(record, feature_columns)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    st.write(f"Estimated price (lakhs): {pred[0]:.2f}")
