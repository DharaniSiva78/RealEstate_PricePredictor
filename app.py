import streamlit as st
import pandas as pd
import joblib


model = joblib.load("enhanced_model.pkl")

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")


st.markdown("""
    <style>
    .stApp, body {
        background-color: #000000;
        color: white;
    }

    
    label, .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stSlider label,
    .css-1d391kg, .css-1v0mbdj, .css-ffhzg2 {
        color: white !important;
    }

    
    .stTextInput > div > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div,
    .stSlider > div {
        background-color: #1a1a1a !important;
        color: white !important;
        border-color: #333333;
    }

    
    .css-1wa3eu0-option {
        color: white !important;
        background-color: #1a1a1a !important;
    }

    
    .stButton>button {
        color: white !important;
        background-color: #333333;
        border: none;
        padding: 0.5em 1em;
        border-radius: 8px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #555555;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Real Estate Price Predictor")
st.markdown("Fill in the house details below to estimate the market price in India.")


with st.form("prediction_form"):
    area = st.number_input("Total Area (sq ft)", min_value=300, max_value=20000, value=1000)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
    stories = st.slider("Number of Stories", 1, 4, 2)
    parking = st.slider("Parking Spots", 0, 4, 1)

    mainroad = st.selectbox("Is it on a main road?", ["yes", "no"])
    guestroom = st.selectbox("Guest Room Available?", ["yes", "no"])
    basement = st.selectbox("Basement Available?", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating?", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning?", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area?", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_data = pd.DataFrame([{
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad,
            'guestroom': guestroom,
            'basement': basement,
            'hotwaterheating': hotwaterheating,
            'airconditioning': airconditioning,
            'prefarea': prefarea,
            'furnishingstatus': furnishingstatus
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"Estimated House Price: ₹ {int(prediction):,}")
