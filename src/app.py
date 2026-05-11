import streamlit as st # type:ignore
import joblib
import pandas as pd
import numpy as np

# Load tools
model = joblib.load('src/housing_model.pkl')
model_columns = joblib.load('src/model_columns.pkl')

st.set_page_config(page_title="Housing Predictor", page_icon="🏠")

st.title("🏡 Smart House Price Predictor")
st.markdown("---")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Living Area (sq ft)", value=1500, step=100)
    quality = st.slider("Overall Quality (1-10)", 1, 10, 6)
    rooms = st.number_input("Total Rooms", value=6, step=1)

with col2:
    year = st.number_input("Year Built", value=2000, min_value=1900, max_value=2026)
    garage = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    # Add a checkbox for a common feature
    is_central_air = st.checkbox("Central Air Conditioning")

if st.button("Calculate Market Value", use_container_width=True):
    # 1. Create a row of 224 zeros
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 2. Map the inputs
    input_df['GrLivArea'] = area
    input_df['OverallQual'] = quality
    input_df['TotRmsAbvGrd'] = rooms
    input_df['YearBuilt'] = year
    input_df['GarageCars'] = garage
    if is_central_air:
        if 'CentralAir_Y' in input_df.columns:
            input_df['CentralAir_Y'] = 1
            
    # 3. Predict
    pred_log = model.predict(input_df)
    price = np.expm1(pred_log)[0]
    
    st.balloons()
    st.metric(label="Estimated Price", value=f"${price:,.2f}")